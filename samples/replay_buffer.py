from collections import deque
import random

class Sample(object):
    def __init__(self):
        pass

# 每一个batch sample格式为
# sample['states'] = ['s1', 's2',...'sn']
# sample['actions'] = ['a1', 'a2',...'an']
# sample['rewards'] = ['r1', 'r2',...'rn']
# sample['next_states'] = ['s'1', 's'2',...'s'n']
# sample['dones'] = ['done1', 'done2',...'donen']
# sample['infos'] = ['info1', 'info2',...'infon']
class ReplayMemory(object):
    def __init__(self, max_count):
        self.memory_queue = deque(maxlen=max_count)

    def push(self, sample):
        self.memory_queue.append(sample)

    def pop(self):
        return  self.memory_queue.pop()

    def samples(self, k):
        zip_samples = {}
        samples = random.sample(self.memory_queue, k)     #返回的是一个包含k个元素的list
        states,actions,rewards,next_states,dones,infos = zip(*samples)      #按元素解包，然后按位置组成元组
        zip_samples['states'] = states
        zip_samples['actions'] = actions
        zip_samples['rewards'] = rewards
        zip_samples['next_states'] = next_states
        zip_samples['dones'] = dones
        zip_samples['infos'] = infos
        return zip_samples

    def size(self):
        return len(self.memory_queue)


if __name__ == '__main__':
    rp = ReplayMemory(60)
    for i in range(1, 100):
        rp.push((i,i*2,i*3,i*4,i*5,i*10))
    print('count number is :', rp.size())

    sap = rp.samples(10)
    print('batch samples is:', sap)
    for _ in range(70):
        if rp.size() >0:
            print(rp.pop())

    print(random.sample('string', 3))
    print(random.sample((1,2,3,4,5), 3))
    # print(random.sample({1,2,3,4,5}, 3)) 集合不是序列类型
