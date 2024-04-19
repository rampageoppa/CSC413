import pickle
import zlib


with open('/home/guxunjia/project/DenseTNT_modified/test1/temp_file/ex_list', 'rb') as handle:
    env = pickle.load(handle)
    instance = pickle.loads(zlib.decompress(env[0]))
    import pdb; pdb.set_trace()