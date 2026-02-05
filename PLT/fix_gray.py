# -*- coding: utf-8 -*-
p = r"D:\Desktop\毕业论文\code\DeepASM-Net\PLT\try.ipynb"
with open(p, "r", encoding="utf-8") as f:
    s = f.read()
s = s.replace("plt.imshow(U0, cmap='gray')", "plt.imshow(U0)")
s = s.replace("plt.imshow(I_z, cmap='gray')", "plt.imshow(I_z)")
with open(p, "w", encoding="utf-8") as f:
    f.write(s)
print("Done")
