import pandas as pd
import time
data = pd.read_csv('complete.csv', delimiter=',').sort_values("awardYear").reset_index()
del data["index"]





from celluloid import Camera
import matplotlib.pyplot as plt

"""fig, ax = plt.subplots()

#  head_length - длина острия стрелки:
ax.arrow(0,1,2, 0,width = 0.02,color="black")

ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
plt.axis("off")
plt.scatter(-0.1,2,edgecolors="black")


plt.show()"""
fig,ax = plt.subplots()
camera = Camera(fig)
for i in range(1900,1929):
    ax.scatter(i, 1, )
    ax.text(i, 1.01, s=i)

    camera.snap()

animation = camera.animate()

plt.plot([1900,2000],[1,1],"r--")
plt.axis("on")
plt.show()