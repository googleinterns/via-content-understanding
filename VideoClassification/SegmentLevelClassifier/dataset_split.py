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
import getopt
import readers
import sys
import writer

if __name__ == "__main__":
  assert len(sys.argv) == 4, ("Incorrect number of arguments {}. Should be 3. Please consult the README.md for proper argument use.".format(len(sys.argv)-1))
  short_options = "i:w:p:"
  long_options = ["input_dir=", "write_dir=", "pipeline_type="]
  try:
    arguments, values = getopt.getopt(sys.argv[1:], short_options, long_options)
  except getopt.error as err:
    print(str(err))
    sys.exit(2)

  for current_argument, current_value in arguments:
    if current_argument in ("-i", "--input_dir"):
      input_dir = current_value
    elif current_argument in ("-w", "--write_dir"):
      write_dir = current_value
    elif current_argument in ("-p", "--pipeline_type"):
      pipeline_type = current_value
  segment_reader = readers.SplitDataset(pipeline_type=pipeline_type)
  input_dataset = segment_reader.get_dataset(input_dir, batch_size=1, type=pipeline_type)
  writer.split_data(write_dir, input_dataset, pipeline_type=pipeline_type)