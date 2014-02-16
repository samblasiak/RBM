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
import basic_rbm

sig=rbm_utils.sigmoid

class GausRBM(basic_rbm.BasicRBM):
    '''
    W - weights
    b1 - bias from the hidden layer to the visible layer
    b2 - bias from the visible layer to the hidden layer
    
    It is possible that this model is incorrect
        -Evidence in favor of correctness
            -magnitude of the gradient is decreases
            -reconstruction is reasonably for normalized data
            -gradient updates are probably correct
            
        -Evidence for incorrectness
            -weight vectors are noisy
            
        -possibly the regularization, precision, or learning rates still need to be adjusted
        -data may need additional normalization
    '''
    
    def __init__(self,data,H,lam=1e-1,gamma=.001,Gprec=1.):
        '''
        data - the data set to run the RBM on, dimensions of the dataset give the number of visible units
        H - number of hidden units
        lam - precision on the Gaussian prior over the parameters
        gamma - the learning rate
        Gprec - precision of the conditional normal distribution over visible units
                    -currently the precision is not used
        '''
        basic_rbm.BasicRBM.__init__(self,data,H,lam,gamma)
        self.Gprec=Gprec
        
    def f(self,x,grad):
        '''
        This computes an approximate gradient of the Half-Gaussian RBM using CD-1
            -similar to the Basic RBM, 
                but samples the visible element from a normal distribution 
        '''
        s=self

        W,b1,b2=s.unpack(x)
    
        #CD-1 step        
        #sample h
        dotW=np.dot(s.data*s.Gprec,W)
        eh=sig(dotW+b2[None,:])
        h=rnd.rand(len(s.data),s.H)<eh
        #sample v
        dotH=np.dot(h,W.T)
        v=rnd.normal(0.,1./np.sqrt(s.Gprec), (len(s.data),s.KK) )+dotH+b1[None,:]
#        v=dotH+b1[None,:]
        #end CD-1 step
    
        dotW1=np.dot(v*s.Gprec,W)
        eh1=sig(dotW1+b2[None,:])
        
        dW=(np.dot(s.data.T,eh)-np.dot(v.T,eh1))*s.Gprec
#        db1=np.sum((s.data-v-b1)*s.Gprec,0)
        db1=np.sum((s.data-v)*s.Gprec,0)
        db2=np.sum(eh-eh1,0)
        
        s.pack(grad,dW,db1,db2)
        grad/=len(s.data)
        grad-=s.lam*x
    
    def cmp_mean_field(self,x):
        s=self
        W,b1,b2=s.unpack(x)
        
        #compute mean field approximation of each patch
        dotW=np.dot(s.data,W)
        eh=sig(dotW+b2[None,:])
        dotH=np.dot(eh,W.T)
        return dotH+b1[None,:]
    
    def plot_images_nonwhite(self,img,x,inv_v):
        s=self
        
        y=s.cmp_mean_field(x)
    
        ndat=np.zeros( img.shape )
        idat=np.zeros( img.shape )
        m=0
        for i in xrange(0,img.shape[0],s.K):
            for j in xrange(0,img.shape[1],s.K):
                ndat[i:i+s.K,j:j+s.K]=np.dot(y[m,:],inv_v).reshape( (s.K,s.K) )
                idat[i:i+s.K,j:j+s.K]=np.dot(s.data[m,:],inv_v).reshape( (s.K,s.K) )
                m+=1

        fig=plt.figure()        
        ax=fig.add_subplot(1,2,1)
        ax.imshow(ndat,interpolation='nearest',cmap=plt.cm.gray)
        ax=fig.add_subplot(1,2,2)
        ax.imshow(idat,interpolation='nearest',cmap=plt.cm.gray)

if __name__=='__main__':
    
    lena=misc.lena()
    lena=rbm_data.get_normed_image(lena,'SIGMOID')
#    lena=rbm_data.get_normed_image(lena)
    data=rbm_data.get_img_patches(lena)
#    data,inv_v=rbm_data.whiten(data)
    data=rbm_data.norm_data(data)
    
    rbm=GausRBM(data,32)
    x=rbm.run(2000,LOAD=False)
    rbm.plot_weights(x)
#    rbm.plot_images_nonwhite(lena,x,inv_v)
    rbm.plot_images(lena,x)
    plt.show()
