from common import Tokenizer,ModelCLS,ModelGEN,ModelPPO,generate
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


tokenizer = Tokenizer()
[tokenizer.decode(i) for i in tokenizer.get_batch_data(prefix=True)[1]][:10]

model_ppo = ModelPPO(torch.load('gen.model'))
model_ppo_ref = ModelPPO(torch.load('gen.model'))

for i in model_ppo_ref.parameters():
    i.requires_grad_(False)


def get_kl(a, b):
    method = 'kl'

    if method == 'kl':
        return a - b

    if method == 'abs':
        return (a - b).abs()

    if method == 'mse':
        return (a - b).square() * 0.5

    if method == 'full':
        return torch.nn.functional.kl_div(a,
                                          b,
                                          log_target=True,
                                          reduction='none')


get_kl(torch.randn(15), torch.zeros(15))

from trl.core import clip_by_value, logprobs_from_logits, masked_mean, masked_whiten


class PPOTrainer:

    def __init__(self):
        self.optimizer = torch.optim.Adam(model_ppo.parameters(), lr=1e-5)

    def step(self, question, answer, reward):
        with torch.no_grad():
            #编码
            token = [q.tolist() + a.tolist() for q, a in zip(question, answer)]
            input_ids, attention_mask = tokenizer.batch_pad(token=token)
            del token
            input_ids = torch.LongTensor(input_ids).to(device)
            attention_mask = torch.LongTensor(attention_mask).to(device)

            #question和answer不需要内容,只需要长度信息即可
            lens_q = [len(i) for i in question]
            lens_a = [len(i) for i in answer]
            del question
            del answer

            #根据question计算answer的概率,并计算每个动作的分数
            prob_log, value, mask = self.batched_forward_pass(
                model_ppo, input_ids, attention_mask, lens_q, lens_a)

            #使用ref模型计算概率,这是为了计算kl散度
            prob_log_ref, _, _ = self.batched_forward_pass(
                model_ppo_ref, input_ids, attention_mask, lens_q, lens_a)

            #计算两份概率的kl散度,并融入reward
            reward = self.compute_rewards(reward, prob_log, prob_log_ref, mask)

            #计算delta和target,用于计算loss
            value, delta, target = self.compute_advantages(value, reward, mask)

        #每批数据循环N次模型
        for _ in range(4):
            #每次算一个数据
            for i in range(len(input_ids)):
                #重新计算概率和value
                prob_log_new, value_new, _ = self.batched_forward_pass(
                    model_ppo, input_ids[i].unsqueeze(0),
                    attention_mask[i].unsqueeze(0), [lens_q[i]], [lens_a[i]])

                #根据新旧概率求出变化率,进而求出loss
                #根据target和value的差可以计算出另外一份loss
                loss = self.get_loss(prob_log[i].unsqueeze(0),
                                     value[i].unsqueeze(0), prob_log_new,
                                     value_new, mask[i].unsqueeze(0),
                                     delta[i].unsqueeze(0),
                                     target[i].unsqueeze(0))

                if not loss:
                    continue

                loss.backward()
                #torch.nn.utils.clip_grad_norm_(model_ppo.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

    def batched_forward_pass(self, model, input_ids, attention_mask, lens_q,
                             lens_a):
        logits, value = model(input_ids=input_ids,
                              attention_mask=attention_mask)

        #取每个字的概率对数
        prob_log = logprobs_from_logits(logits[:, :-1], input_ids[:, 1:])

        #是预测结果并且不是PAD的位置是1
        mask = torch.zeros_like(attention_mask)
        mask[:, :-1] = attention_mask[:, 1:]
        for i in range(len(input_ids)):
            start = lens_q[i] - 1
            end = start + lens_a[i]
            mask[i, :start] = 0
            mask[i, end:] = 0

        #对最后一个字的预测没有意义,直接丢弃
        value = value[:, :-1]
        mask = mask[:, :-1]

        return prob_log, value, mask

    def compute_rewards(self, reward, prob_log, prob_log_ref, mask):
        reward_kl = []

        for i in range(len(reward)):
            #求两份概率的kl散度
            kl = get_kl(prob_log[i], prob_log_ref[i]) * -0.2

            #把reward加在最后一个字的kl散度上
            if (mask[i] == 0).all():
                #print('all 0')
                idx = 0
            else:
                idx = mask[i].nonzero()[-1].item()
            kl[idx] += reward[i]

            reward_kl.append(kl)

        return torch.stack(reward_kl)

    def compute_advantages(self, value, reward_kl, mask):
        value = value * mask
        reward_kl = reward_kl * mask

        delta = []
        lens = reward_kl.shape[1]

        #从后往前遍历
        for i in reversed(range(lens)):
            #取下一时刻的value,如果已经是最后一个时刻,则value_next是0
            #因为整个循环是从后往前,所以第0次是0,其他时刻取value
            value_next = 0
            if i < lens - 1:
                value_next = value[:, i + 1]

            #value = gamma*下一时刻的value + reward
            #理论上相等,这里的差定义为delta,这里gamma是1,所以省略了
            d = reward_kl[:, i] + value_next - value[:, i]

            #取最后一个delta,如果还没有,则初始化为0
            last_d = 0
            if delta:
                last_d = delta[-1]

            #delta是从后往前传递的,这里的系数衡量了前后动作的因果关联性
            delta.append(d + 0.95 * last_d)

        #翻转顺序
        delta = torch.stack(delta[::-1]).transpose(0, 1)

        #定义target,它估计了理想的value值
        target = delta + value
        delta = masked_whiten(delta, mask)

        return value, delta, target

    def get_loss(self, prob_log, value, prob_log_new, value_new, mask, delta,
                 target):

        #对数概率,相除变相减,取exp后还原为商,即两个模型输出logits的变化率
        ratio = (prob_log_new - prob_log).exp()

        #如果变化率太过于剧烈,可能是发生了震荡,跳过
        if masked_mean(ratio, mask).item() > 10:
            #print('skip', masked_mean(ratio, mask).item())
            return None

        #先算两个value的loss,简单的算mse loss就可以了
        loss_vf1 = (value_new - target)**2
        #数值裁剪,很显然是为了缓解自举
        loss_vf2 = clip_by_value(value_new, value - 0.2, value + 0.2)
        loss_vf2 = (loss_vf2 - target)**2
        #两份loss取大的,还是为了缓解自举
        loss_vf = 0.5 * masked_mean(torch.max(loss_vf1, loss_vf2), mask)

        #计算ppo loss
        loss_surr1 = -delta * ratio
        #数值裁剪,很显然是为了缓解自举
        loss_surr2 = -delta * ratio.clamp(0.8, 1.2)
        loss_surr = masked_mean(torch.max(loss_surr1, loss_surr2), mask)

        return loss_surr + 0.1 * loss_vf


trainer = PPOTrainer()

trainer

model_cls = torch.load('cls.model')
model_cls.to(device)

for i in model_cls.parameters():
    i.requires_grad_(False)

def get_question():
    label, question, _ = tokenizer.get_batch_data(prefix=True)
    label = torch.LongTensor(label).to(device)

    #只要问题部分,等号后面的内容切除
    question = [i[:i.index(tokenizer.encoder['=']) + 1] for i in question]
    question = [torch.LongTensor(i).to(device) for i in question]

    return label, question


label, question = get_question()

label, question[:10]

#如果question的长度确定,这里可以转换成批运算
def get_answer(question):
    answer = [generate(model_ppo.model_gen, i.unsqueeze(0)) for i in question]

    #裁剪,只要生成的部分
    answer = [a[0, len(q):] for q, a in zip(question, answer)]

    return answer


answer = get_answer(question)

answer[:10]

def get_reward(question, answer, label):
    token = [q.tolist() + a.tolist() for q, a in zip(question, answer)]

    input_ids, attention_mask = tokenizer.batch_pad(token=token)
    input_ids = torch.LongTensor(input_ids).to(device)
    attention_mask = torch.LongTensor(attention_mask).to(device)

    with torch.no_grad():
        logits = model_cls(input_ids=input_ids, attention_mask=attention_mask)

    return logits.gather(1, label.reshape(-1, 1)).squeeze(1)


reward = get_reward(question, answer, label)

reward

for epoch in range(2000):
    label, question = get_question()
    answer = get_answer(question)
    reward = get_reward(question, answer, label)

    trainer.step(question, answer, reward)

    if epoch % 100 == 0:
        print(epoch, reward.mean().item())
        for _, q, a, r in zip(range(2), question, answer, reward):
            q = tokenizer.decode(q.tolist())
            a = tokenizer.decode(a.tolist())
            r = r.item()
            print(q, a, r)

model_ppo.to('cpu')
torch.save(model_ppo, 'ppo.model')
