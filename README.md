# Signal Propagation in spike and rate-model of neurons

This repository contains large scale neural network (not artificial) simulation (synthetic) using

<ol>
  <li> Firing rate model  </li>
  <li> Spiking rate model  </li>
  </ol>
  
For the rate model, simulations are carried out using Python and for spike model via Brian2 python module (https://github.com/brian-team/brian2/).

 
We aim to replicate the finding in Inter-areal balanced amplification enhances signal propagation in a large-scale circuit model of the Primate Cortex (https://www.sciencedirect.com/science/article/pii/S0896627318301521#mmc1) that uses the Brian2 module.


The anotomical connectivity of the *Macaque*, incorporated in our simulations, is obtained from http://core-nets.org/index.php (Database section).
Also, available in the Excel format here: (https://ars.els-cdn.com/content/image/1-s2.0-S0896627315007655-mmc2.xlsx)

The anatomical connectivity is defined in terms of *Fraction of Labelled Neurons* as

FLN_{B-to-A} is the ratio of (# neurons projecting to area A from area B) to the  (Total # neurons projecting to area A from all areas) 
 

