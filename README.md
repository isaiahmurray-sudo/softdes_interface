# Resin Poll Data Collector

## Highlights
* Integration of sensor readings from several devices
* Easy data collector GUI to set up polls
* Debug mode to remotely test changes

## Overview
This code is part of a greater project at Formlabs. It serves to aid in the dielectric analysis (DEA) workflow
by running tests and creating a framework to store and manipulate data. This repo is a subsection of the project's
primary repository with any sensative information omitted.

<p>
DEA is one tool to analyze how a chemical responds to a strimuli, in our case UV light. An interdigiated comb gets an AC current from the LCR meter which in turn makes a changing magnetic field. Modules in the resin respond to that field and their movement (or lack thereof) is reflected in changes to capacitance and impedence. It lets us get a peek into what may be happening chemically during a 3D print.

**Important terminology**
* ADC -> Analog to digitial converter (connected to a photoresistor)
* LCR -> Type of instrument that measures inductance, capacitance, and resistance
* Leash -> Dubug interface for the printer
* *Poll* -> A single test where we form one physical sample and record the instrument readings throughout
* Ionic viscosity -> (in theory) how much the molecules in the chemical resist moving


## Installation and Execution
``` bash
pip install -r requirements.txt

python poll_data_collector.py
```