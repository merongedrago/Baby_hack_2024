# Meron_Gedrago_miniproject1
[![CI](https://github.com/nogibjj/Meron_Gedrago_miniproject1/actions/workflows/hello.yml/badge.svg)](https://github.com/nogibjj/Meron_Gedrago_miniproject1/actions/workflows/hello.yml)

Welcome to the first project for Data Engineering!

This project aims to create a python github template that can be reused in the future and also get us to start understanding best practices in creating/sharing/replicating code. The components of this template should allow for anyone else to get this project/code and be able to reproduce it without any problem on their local computer. It contains the following element to enable that   

* *README.md*: this is the document that you are reading right now that explains the purpose of the project and any important information to the reader or whoever wants to replicate the project 
* *requirements.txt*: list of packages that the project uses/"requires" (as the name suggests) with the version numbers 
* *MG_main.py*: a simple multiplication function that inputs two numbers and multiplies them!
* *Test_MG_main.py*: a series of cases that test whether or not my MG_main code works correctly 
* *Makefile*: series of useful commands that we want to run when we call it such as installing the required packages from the requirements document, formatting the code, running our test and other commands we would want to automate 
* *.devcontainer*: this enables the setting up of a virtual environment that will allow the code to work on any computer and not just on the one that it was created on. This file contains the devcontainer and Dockerfile, which are long templatized codes and are changed slightly across projects
* *CI/CD or hello.yml*: a set of instructions or automated actions that we want to be executed after a certain action e.g., when the code is pushed to github or pulled  
