'''
    Small Scale Restricted Bolzmann Machine Implementation
    Copyright (C) 2014 Samuel J. Blasiak

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
'''


from scipy import misc
import scipy.linalg as lin
import numpy as np
import numpy.random as rnd
import pylab as plt
import scipy.io as sio
import rbm_utils
import rbm_data

sig=rbm_utils.sigmoid

class BasicRBM:
    '''
    W - weights
    b1 - bias from the hidden layer to the visible layer
    b2 - bias from the visible layer to the hidden layer 
    '''
    
    def __init__(self,data,H,lam=1e-3,gamma=.5):
        '''
        data - the data set to run the RBM on, dimensions of the dataset give the number of visible units
        H - number of hidden units
        lam - precision on the Gaussian prior over the parameters
        gamma - the learning rate 
        '''
        s=self
        
        s.H=H
        s.lam=lam
        s.gamma=gamma
    
        s.data=data
        s.N,s.KK=s.data.shape
        s.K=int(np.sqrt(s.KK))
    
    def unpack(self, x):
        s=self
        W=np.reshape(x[:s.H*s.KK], (s.KK,s.H))
        b1=x[-s.H-s.KK:-s.H]
        b2=x[-s.H:]
        return W,b1,b2
    
    def pack(self,x,W,b1,b2):
        s=self
        x[:s.H*s.KK]=W.reshape( (s.H*s.KK) )
        x[-s.H-s.KK:-s.H]=b1
        x[-s.H:]=b2
        
    def f(self,x,grad):
        s=self
        W,b1,b2=s.unpack(x)
    
        #contrastive divergence with one sampling iteration
        #multiple sampling iterations
    #    for i in xrange(max(1,int(np.log(iter)/np.log(10.)))):
            
        #sample h
        dotW=np.dot(s.data,W)
        eh=sig(dotW+b2[None,:])
        h=rnd.rand(len(s.data),s.H)<eh
        #sample v
        dotH=np.dot(h,W.T)
        v=rnd.rand(len(s.data),s.KK)<sig(dotH+b1[None,:])
        
        dotW1=np.dot(v,W)
        eh1=sig(dotW1+b2[None,:])
            
        dW=np.dot(s.data.T,eh)-np.dot(v.T,eh1)
        db1=np.sum(s.data-v,0)
        db2=np.sum(eh-eh1,0)
        
        s.pack(grad,dW,db1,db2)
        grad/=len(s.data)
        grad-=s.lam*x
    
        
    def run(self, max_iters, LOAD=False, SAVE=True, seed=0):
        s=self
        rnd.seed(seed)

        if LOAD:
            dat=sio.loadmat(str(self.__class__)[9:])
            x=dat['RBM']
        else:
            #initialize the RBM paramters
            x=rnd.rand(s.H*s.KK+s.H+s.KK)
            rn=np.sqrt(6./(s.H+s.KK+1))
            x[:s.H*s.KK]*=2*rn
            x[:s.H*s.KK]-=rn
            rn=np.sqrt(6./(s.H+1))
            x[-s.H-s.KK:]=0

        #intialize the array holding the gradient
        grad=np.zeros(s.H*s.KK+s.H+s.KK)
    
        for iter in xrange(1,max_iters+1):
            s.f(x,grad)
            x+=s.gamma*grad
            print '%5d%10.3f'%(iter,lin.norm(grad))
    
        #minibatch RBM iterations
    #    _data=data
    #    batch=100
    #    for iter in xrange(1,100):
    #        for n in xrange(0,len(data),batch):
    #            data=_data[n:n+batch]
    #            x+=gamma*f(x)
    #            gamma*=.999
    #        data=_data[n:]
    #        x+=gamma*f(x)
    #    data=_data    
        
        if SAVE:
            sio.savemat(str(self.__class__)[9:], {'RBM':x})
            
        return x
        

    def plot_weights(self,x):
        '''
        plot the inputs that maximize the probability of the RBM 
            for each individual weight
        '''
        s=self
        W,b1,b2=s.unpack(x)
        
        fig=plt.figure()
        for i in xrange(s.H):
            D1=np.ceil(np.sqrt(s.H))
            D2=np.ceil(s.H/float(D1))
            ax=fig.add_subplot(D1,D2,i+1)
    #            ax.imshow(data[resp_mx[i]].reshape( (K,K) ),interpolation='nearest',cmap=plt.cm.gray)
            ax.imshow(((W[:,i]-np.average(W[:,i]))/lin.norm(W[:,i])).reshape( (s.K,s.K) ),interpolation='nearest',cmap=plt.cm.gray)

    def cmp_sqerr(self,x):
        y=self.cmp_mean_field(x)
        return np.sum((self.data-y)**2)
        
    def cmp_mean_field(self,x):
        s=self
        W,b1,b2=s.unpack(x)
        
        #compute mean field approximation of each patch
        dotW=np.dot(s.data,W)
        eh=sig(dotW+b2[None,:])
        dotH=np.dot(eh,W.T)
        return sig(dotH+b1[None,:])
        
        
    def plot_images(self,img,x):
        s=self
        
        y=s.cmp_mean_field(x)
    
        ndat=np.zeros( img.shape )
        idat=np.zeros( img.shape )
        m=0
        for i in xrange(0,img.shape[0],s.K):
            for j in xrange(0,img.shape[1],s.K):
                ndat[i:i+s.K,j:j+s.K]=y[m,:].reshape( (s.K,s.K) )
                idat[i:i+s.K,j:j+s.K]=s.data[m,:].reshape( (s.K,s.K) )
                m+=1

        fig=plt.figure()        
        ax=fig.add_subplot(1,2,1)
        ax.imshow(ndat,interpolation='nearest',cmap=plt.cm.gray)
        ax=fig.add_subplot(1,2,2)
        ax.imshow(idat,interpolation='nearest',cmap=plt.cm.gray)
                
        

if __name__=='__main__':
    
    lena=misc.lena()
    lena=rbm_data.get_normed_image(lena,'BINARY')
    data=rbm_data.get_img_patches(lena)
    
    rbm=BasicRBM(data,32)
    x=rbm.run(500)
    rbm.plot_weights(x)
    rbm.plot_images(lena,x)
    plt.show()