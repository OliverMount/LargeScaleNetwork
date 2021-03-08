 
# Neural Signal Propagation Framework
 

This post contains large scale neural network (not artificial) simulation (synthetic) using

<ol>
  <li> Firing rate model  </li>
  <li> Spiking rate model  </li>
  </ol>
  
All the simulations are carried out via Brian2 python module ( https://github.com/brian-team/brian2/ ). 
The anotomical connectivity of the *Macaque*, incorporated in our simulations, is obtained from http://core-nets.org/index.php (Database section).
Also, available in the Excel format here: (https://ars.els-cdn.com/content/image/1-s2.0-S0896627315007655-mmc2.xlsx)

The anatomical connectivity is defined in terms of *Fraction of Labelled Neurons* as

FLN_{B-to-A} is the ratio of (# neurons projecting to area A from area B) to the  (Total # neurons projecting to area A from all areas) 
 
We aim to replicate the finding in Inter-areal Balanced Amplification Enhances Signal Propagation in a Large-Scale Circuit Model of the Primate Cortex (https://www.sciencedirect.com/science/article/pii/S0896627318301521#mmc1) that uses the Brian2 module.







