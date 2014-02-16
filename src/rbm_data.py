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


import scipy.linalg as lin
import numpy as np

def get_normed_image(img,norm_opt=None):
    '''
    img - an input image
    norm_opt - how to normalize the image data
    '''
    
    l=np.zeros(img.shape,dtype=np.float)
    
    l[:,:]=img-np.average(img)
    
    if norm_opt!=None:
        if norm_opt=='BINARY':
            print 'Binary normalization'
            l=l>0
            
        elif norm_opt=='SIGMOID':
            
            st=3*np.std(l)
            l[l>st]=st
            l[l<-st]=-st
            
            l/=2*st
        
            l+=1;l*=.4;l+=.1
    
    print '--get_normed_image--'
    print 'min value: ',np.min(l)
    print 'max value: ',np.max(l)
    print 
    
    return l

'''
if TEST:
    K=2
    data=np.zeros( (3,K*K) )
else:


if TEST:
    for m in xrange(len(data)):
        i=rnd.randint(l.shape[0]-K+1)
        j=rnd.randint(l.shape[1]-K+1)
        data[m,:]=l[i:i+K,j:j+K].reshape( K*K )
else:
'''

def get_img_patches(img,K=8):
    '''
    img - the input image
    K - the width and height of the patch
    '''
    
    data=np.zeros( ((img.shape[0]/K)*(img.shape[1]/K),K*K) )

    #N is the number of examples, KK is the number of visible units, 
    N,KK=data.shape
    print '--get_img_patched--'
    print 'num patches:',N
    print 'patch width:', K
    print 'total patch size:',KK

    m=0
    for i in xrange(0,img.shape[0],K):
        for j in xrange(0,img.shape[1],K):
            data[m,:]=img[i:i+K,j:j+K].reshape( K*K )
            m+=1
    
    return data
    
def norm_data(data):
    mn=np.mean(data,0)
    data-=mn
    st=np.std(data,0)
    data/=st
    
    return data
    
    
def whiten(data):
    cv=np.dot(data.T,data)
    e,v=lin.eig(cv)
    v=np.real(v)
    e=np.real(e)
    inv_e=np.diag(np.sqrt(e))
    e=np.diag(1./np.sqrt(e))
    
    u=np.dot(e,v.T)
    
    white=np.dot(data,u.T)
    inv_v=np.dot(inv_e,v.T)
    
    assert np.all( ( np.dot(white.T,white)-np.eye(data.shape[1])) < 1e-5 )
    assert np.all( (np.dot(u.T,inv_v)-np.eye(data.shape[1])) < 1e-5 )
    return white,inv_v
