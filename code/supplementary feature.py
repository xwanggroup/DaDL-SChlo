#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import csv

num = 200
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
print(generator)
state_dict = torch.load('./gen_5000.pt')
generator.load_state_dict(state_dict)
f = open('results.csv', 'w', encoding='utf-8', newline="")
csv_write = csv.writer(f)
for i in range(num):
    rand = 2 * (torch.rand(1, 128, device=device) - 0.5)
    one_dimension = generator(rand).cpu().detach().numpy()[0].tolist()
    one_dimension = [str(i) for i in one_dimension]
    csv_write.writerow(one_dimension)

