import torch

#Let's create a tensor (basically a storage of numbers that can be any dimension, etc.)

torch.tensor([1,2,3])
torch.arange(1,11)

#You can also create a tensor with randomized numbers (very important for training neural networks as they shift random data w/ dataset)
# the numbers will be between 0 and 1
torch.rand(3,4)


#Tensor datatypes and receiving information from tensors

some_tensor = torch.rand(2,2)
print(some_tensor.dtype, some_tensor.device, some_tensor.shape, some_tensor.ndim)
#This will print the tensor's datatype (default is float32), device it runs on (cpu/gpu), the tensor's shape (ex: 2x2), and its dimensions


#You can also set the tensor's datatype and other information in case there are errors:
a = torch.arange(1,6, dtype=torch.float16)
print(a.dtype)

#Tensor operations (addition, subtraction, multiplication, division)
tensor = torch.arange(0,10)
tensor + 10
print(tensor)

tensor += 10
tensor -= 10
tensor *= 10
tensor = tensor/10

# Shapes need to be in the right way  
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)

#torch.matmul(tensor_A, tensor_B) # (this will error)

tensor_B = torch.transpose(tensor_B,dim0=0,dim1=1)
#print(tensor_B)
#print(torch.mm(tensor_A,tensor_B))


#Tensor aggregation (min,max,mean,avg) (mean requires tensor of type float32)

t = torch.arange(0,100,10)
print(torch.min(t), t.min())
print(torch.max(t), t.max())
print(torch.mean(t.type(torch.float32)), t.type(torch.float32).mean())
print(torch.sum(t),t.sum())


#Positional min and max (the index at which the min/max is at)
torch.argmin(t)
torch.argmax(t)

#Reshaping,stacking,squeezing,and unsqueezing tensors
reshaped = tensor.reshape(10,1)
print(reshaped,tensor.shape,reshaped.shape)
print(reshaped.view(5,2)) #this changes the memory of the input tensor as well

stacked = torch.stack([tensor,tensor,tensor,tensor], dim=0)
print(stacked)

squeezed = reshaped.squeeze()
print(squeezed,reshaped.shape,squeezed.shape)

unsqueezed = squeezed.unsqueeze(dim=0)
print(unsqueezed,squeezed.shape,unsqueezed.shape)

#Permuting tensors
my_tensor = torch.rand(size=(224,224,3))
permuted = my_tensor.permute(2,0,1)
print(my_tensor,permuted)