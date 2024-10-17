# https://stackoverflow.com/questions/1685221/accurately-measure-time-python-function-takes

from time import process_time

t = process_time()
#do some stuff
elapsed_time = process_time() - t