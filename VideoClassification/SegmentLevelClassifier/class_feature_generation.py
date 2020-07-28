"""Copyright 2020 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Compute and add Class specific features to the data.
"""
import numpy as np
import os
import pandas as pd
import tensorflow as tf

#1. Split data into segments
# A.Includes making reader which get segments.
# B.Code that will then split segments. Need to get data on a segment basis.
#2. (Segment, cand_class) pairs. Will loop through each tuple and get 

#Use previous reader, get segments and then shoot them out.
#Said segments are already labelled.
#Retain metadata bc CSF are not compared within same video.
#Compile list of train segments per class. Each segment has 5760 ints. 273k segments split among all classes.
#Use said list to calculate CSF.
#Else can just loop through all data but this seems bad.

#Generate class specific features for both train and test.
#If we have 1000 files. Then have a dataset of all segments
#Make a file which splits data into segments and also stores data into 1000 class files.
#CSF Generation will then loop through dataset of all segments and for each one, it will look at the class chosen
#for it, loop through said class file and get distance between segment and all others except ones that match the video id.
#CSF will store said data and then add it in shards