from CreateDataFrame import *

df = readData('data.txt')
dict = createDictionary(df)
languages = getUniqueLanguages(df)
features = getUniqueFeatures(df)

print(len(features))
print(len(languages))

def find_next_pair(index1, index2, last_index):
  if index2 < last_index:
    index2 = index2 + 1
    return index1, index2
  else:
    if index1 < last_index - 1:
      index1 = index1 + 1
      index2 = index1 + 1
      return index1, index2
    else:
      return None, None

def find_nulls(array1, array2):
  result = array1 + array2
  result = np.invert(np.isnan(result))
  return result.astype(int)

def make_arrays():
  input_array = []
  output_array = []
  null_layer = []
  last_index = len(languages) - 1
  index1 = 0
  index2 = 1
  end = False
  while not end:
    values1 = dict[languages[index1]]['values']
    values2 = dict[languages[index2]]['values']
    nulls = find_nulls(values1, values2)
    input_array.append(values1)
    output_array.append(values2)
    null_layer.append(nulls)
    index1, index2 = find_next_pair(index1, index2, last_index)
    if index1 == None and index2 == None:
      end = True
      input_array = np.array(input_array)
      output_array = np.array(output_array)
      null_layer = np.array(null_layer)
      input_array[np.isnan(input_array)] = 0
      output_array[np.isnan(output_array)] = 0
      return input_array, output_array, null_layer

def pickle_arrays():
  input_array, output_array, null_layer = make_arrays()
  np.save('input_array.npy', input_array)
  np.save('output_array.npy', output_array)
  np.save('null_layer.npy', null_layer)

# pickle_arrays()

# print(input_array[0])
# print(output_array[0])
# print(null_layer[0])
# print(np.shape(input_array))
# print(np.shape(output_array))
# print(np.shape(null_layer))


  
  