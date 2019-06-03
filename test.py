import matplotlib.pyplot as plt
import numpy as np

line = plt.plot([1,2,3,4,5,6,7], [1,4,9,6,13,14,6])
# plt.xlim(0,8)
plt.xticks(np.arange(0, 8, 1.0))
for xy in zip(range(1,8),[1,4,9,6,13,14,6]):
    plt.annotate(str(xy[1]), xy=(xy[0],xy[1]+1), textcoords='data')
plt.setp(line, color='g', linewidth=2.0)
plt.xlabel('number of generation')
plt.ylabel('best accuracy rate among population (%)')
plt.title('best DNA performance over time')
plt.show()