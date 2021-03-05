 
# Neural Signal Propagation Framework
 

This post contains large scale neural network (not artificial) simulation (synthetic) using

<ol>
  <li> Firing rate model  </li>
  <li> Spiking rate model  </li>
  </ol>
  
All the simulations are carried out via Brian2 python module ( https://github.com/brian-team/brian2/ ). 
The anotomical connectivity of the *Macaque*, incorporated in our simulations, is obtained from http://core-nets.org/index.php (Database section).

The anatomical connectivity is defined in terms of *Fraction of Labelled Neurons* as

```math
FLN_{B\simA} = \frac{\sigma}{\sqrt{n}}
```

![img] (http://latex.codecogs.com/svg.latex?%7BB%5CsimA%7D+%3D+%5Cfrac%7B%23+Neurons+projecting+to+area+A+from+area+B%7D%7BTotal+neurons+projecting+to+area+A+from+all+areas%7D+%0D%0A
 
 
$ 
FLN_{B\simA} = \frac{# Neurons projecting to area A from area B}{Total neurons projecting to area A from all areas} 
 $
 


We aim to replicate the finding in Inter-areal Balanced Amplification Enhances Signal Propagation in a Large-Scale Circuit Model of the Primate Cortex (https://www.sciencedirect.com/science/article/pii/S0896627318301521#mmc1) that uses the Brian2 module.





