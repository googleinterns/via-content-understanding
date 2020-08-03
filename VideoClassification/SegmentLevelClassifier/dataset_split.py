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

Split dataset into segments used for Class Feature Generation.
"""
import readers
import writer

if __name__ == "__main__":
  segment_reader = readers.SplitDataset(pipeline_type="test")
  input_dataset = segment_reader.get_dataset("/home/conorfvedova_google_com/data/segments/candidate_test", batch_size=1, type="test")
  writer.split_data("/home/conorfvedova_google_com/data/segments/split_test", input_dataset, pipeline_type="test")

#Cand gen and get labels. Split data into segments but save them however. Not necessarily by segment_label. Only save segments we have an output for.
#Load segments and simply predict for every candidate generation example. Will most likely simply pass in cand_gen list and CSF lists as well and test on that.
#Need to modify CSF so that it runs for multiple classes.