import torch
import torch.nn as nn

class L_con(nn.Module):
	"""docstring for L_con"""
	def __init__(self,eta):
		super(L_con, self).__init__()
		self.eta=eta

		self.down=nn.Sequential(
			nn.AvgPool2d(3,2,1),
			nn.AvgPool2d(3,2,1))

	def forward(self,G,v,i):
		I=torch.pow(torch.pow((self.down(G)-i),2).sum(),0.5)
		r=G-v
		[W,H]=r.shape[2:4]
		tv1=torch.pow((r[:,:,1:,:]-r[:,:,:H-1,:]),2).mean()
		tv2=torch.pow((r[:,:,:,1:]-r[:,:,:,:W-1]),2).mean()
		V=tv1+tv2
		return (I+self.eta*V).mean()

class L_adv_G(nn.Module):
	"""docstring for L_adv_G"""
	def __init__(self):
		super(L_adv_G, self).__init__()
		
	def forward(self,dv,di):
		return torch.log(1-dv*0.5).mean()+torch.log(1-di*0.5).mean()

class L_G(nn.Module):
	"""docstring for L_G"""
	def __init__(self, lam=0.5, eta=1.2):
		super(L_G, self).__init__()
		self.lam=lam
		self.eta=eta

		self.L_con=L_con(self.eta)
		self.L_adv_G=L_adv_G()

	def forward(self,v,i,G,dv,di):
		return self.L_adv_G(dv,di)+self.lam*self.L_con(G,v,i)

class L_Dv(nn.Module):
	"""docstring for L_Dv"""
	def __init__(self):
		super(L_Dv, self).__init__()
	
	def  forward(self,dv_v,dv_G):
		return (-torch.log(dv_v)).mean()+(-torch.log(1-dv_G)).mean()

class L_Di(nn.Module):
	"""docstring for L_Di"""
	def __init__(self):
		super(L_Di, self).__init__()
	
	def  forward(self,di_i,di_G):
		return (-torch.log(di_i)).mean()+(-torch.log(1-di_G)).mean()

if __name__=='__main__':
	vis=torch.rand((1,1,256,256))
	ir=torch.rand((1,1,64,64))
	output=torch.rand((1,1,256,256))
	dv=torch.rand((1,1))
	di=torch.rand((1,1))
	loss=L_Di()
	output=loss(dv,di)
	print(output)