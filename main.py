%matplotlib inline
from scipy import misc
import matplotlib.pyplot as plt
import wave
import numpy as np
import sklearn.preprocessing as pp
ss = pp.StandardScaler()

pixels = 32 * 32
array = ['chrome.png', 'facebook.png', 'google.png', 'pinterest.png', 'twitter.png']
pics = []

for x in array:
    pics.append(misc.imread(x, mode='RGB'))

r_arr = [[] for x in np.zeros(32*32)]
g_arr = [[] for x in np.zeros(32*32)]
b_arr = [[] for x in np.zeros(32*32)]


for i in range(len(pics)):
    idx = 0
    for j in pics[i]:
        for k in j:
            r_arr[idx].append(k[0])
            g_arr[idx].append(k[1])
            b_arr[idx].append(k[2])
            idx += 1

print('R:')
print(np.array(r_arr))
print()
print('G:')
print(np.array(g_arr))
print()
print('B:')
print(np.array(b_arr))

r_arr_avg = np.array([np.average(x) for x in r_arr]).reshape(32,32)
g_arr_avg = np.array([np.average(x) for x in g_arr]).reshape(32,32)
b_arr_avg = np.array([np.average(x) for x in b_arr]).reshape(32,32)

r_arr_var = np.array([np.var(x) for x in r_arr]).reshape(32,32)
g_arr_var = np.array([np.var(x) for x in g_arr]).reshape(32,32)
b_arr_var = np.array([np.var(x) for x in b_arr]).reshape(32,32)


print()
print("Piros átlagból előállított adathalmaz:")
r_arr_avg_final = ss.fit_transform(X=r_arr_avg)
print(r_arr_avg_final)
print()
print("Zöld átlagból előállított adathalmaz:")
g_arr_avg_final = ss.fit_transform(X=g_arr_avg)
print(g_arr_avg_final)
print()
print("Kék átlagból előállított adathalmaz:")
b_arr_avg_final = ss.fit_transform(X=b_arr_avg)
print(b_arr_avg_final)

print()
print("Piros szórásból előállított adathalmaz:")
r_arr_v_final = ss.fit_transform(X=r_arr_var)
print(r_arr_v_final)
print()
print("Zöld szórásból előállított adathalmaz:")
g_arr_v_final = ss.fit_transform(X=g_arr_var)
print(g_arr_v_final)
print()
print("Kék szórásból előállított adathalmaz:")
b_arr_v_final = ss.fit_transform(X=b_arr_var)
print(b_arr_v_final)


waveFile = wave.open('cartoon.wav', 'r')
signal = waveFile.readframes(-1)
signal = np.fromstring(signal,'Int16')

plt.figure(1)
plt.title('Spectogram')
plt.specgram(signal)
plt.show()

ascii_letters = 'abcdefghijklmnopqrstuvwxyz'
dict = {}
for x in ascii_letters:
    dict[x] = 0

with open('text.txt') as f:
    while True:
        c = f.read(1)
        if c.lower() in ascii_letters:
            if not c:
                break
            else:
                dict[c.lower()] += 1


plt.figure(2)
plt.title('Histogram')
plt.bar(np.arange(len(dict)), dict.values(),  align="center", width=0.5)
plt.xticks(np.arange(len(dict)), dict.keys())
y = max(dict.values()) + 1
plt.ylim(0, y)
plt.show()
