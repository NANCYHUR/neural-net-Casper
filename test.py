import matplotlib.pyplot as plt

line = plt.plot([1,2,3,4,5,6,7], [1,4,9,6,13,14,6])
plt.setp(line, color='g', linewidth=2.0)
plt.xlabel('number of generation')
plt.ylabel('best accuracy rate among population (%)')
plt.title('best DNA performance over time')
plt.show()