import torch
import torch.nn as nn
from model import mobilenet
import numpy as np
import zstandard as zstd
import sys
import snappy
import time

debug = False
force_sparsify = False

class Optimizer:
    data_size = 4
    sparse_layer_num = 0
    dense_layer_num = 0
    sparse_cpr_time = 0
    rle_cpr_time = 0

    def __init__(self):
        self.mem = 0
        self.sparse_mem = 0
        self.min_sparse_mem = 0
        self.rle_mem = 0
        self.min_rle_mem = 0
        self.zstd_mem1 = 0
        self.zstd_mem2 = 0
        self.zstd_mem3 = 0
        self.cpr_time = 0
        self.snappy_mem = 0

        self.cctx1 = zstd.ZstdCompressor(level=1, write_checksum=True, threads=-1)
        self.cctx2 = zstd.ZstdCompressor(level=3, write_checksum=True, threads=0)
        self.cctx3 = zstd.ZstdCompressor(level=5, write_checksum=True, threads=0)
        self.dctx = zstd.ZstdDecompressor()

        # self.dropout = nn.Dropout(p=0.9)


    def reset(self):
        self.mem = 0
        self.sparse_mem = 0
        self.min_sparse_mem = 0
        self.rle_mem = 0
        self.min_rle_mem = 0
        self.zstd_mem1 = 0
        self.zstd_mem2 = 0
        self.zstd_mem3 = 0
        self.snappy_mem = 0
        self.cpr_time = 0

    def calculate(self):
        print ( self.cctx1.memory_size(),  self.cctx2.memory_size(), self.cctx3.memory_size())
        return {
            "org_mem" : self.mem ,
            "sparse_mem" : self.sparse_mem,
            "min_sparse_mem" : self.min_sparse_mem,
            "rle_mem" : self.rle_mem,
            "min_rle_mem" : self.min_rle_mem,
            "zstd1" : self.zstd_mem1 + self.cctx1.memory_size(),
            "zstd2": self.zstd_mem2 + self.cctx2.memory_size(),
            "zstd3": self.zstd_mem3 + self.cctx3.memory_size(),
            "snappy" : self.snappy_mem,
            "cpr_time" : self.cpr_time
        }

    def hook(self, module, fea_in, fea_out):
        start = time.time()
        if force_sparsify:
            fea_out = self.dropout(fea_out)


        fea_out = (fea_out > 0)

        org_mem = self.org_mem(fea_out)
        # sparse_mem = self.sparse_compress(fea_out, get_min=False)
        sparse_mem = 0
        sparse_min_mem = self.sparse_compress(fea_out, get_min=True)
        # sparse_min_mem = 0
        # rle_mem = self.rle_compress(fea_out)
        rle_mem = 0

        #zstd_mem1, zstd_mem2, zstd_mem3 = self.zstd_compress(fea_out)
        #snappy_mem = self.snappy_compress(fea_out)
        zstd_mem1, zstd_mem2, zstd_mem3 = 0, 0, 0
        snappy_mem = 0
        end = time.time()



        self.mem += org_mem
        self.sparse_mem += sparse_mem
        self.min_sparse_mem += sparse_min_mem
        self.rle_mem += rle_mem
        self.min_rle_mem += min(rle_mem, org_mem)
        self.zstd_mem1 += zstd_mem1
        self.zstd_mem2 += zstd_mem2
        self.zstd_mem3 += zstd_mem3
        self.snappy_mem += snappy_mem
        self.cpr_time += end - start

        return None

    def register(self, net):
        if isinstance(net, nn.ReLU):
            net.register_forward_hook(hook=self.hook)

        net_children = net.children()
        for child in net_children:
            self.register(child)


    #################################### Memory Calculation Related Functions #############################

    def org_mem(self, x):
        return np.prod(x.shape) * Optimizer.data_size

    def zstd_compress(self, x):
        data = x.detach().numpy()
        compressed_x1 = self.cctx1.compress(data.tobytes())
        # compressed_x2 = self.cctx2.compress(data.tobytes())
        # compressed_x3 = self.cctx3.compress(data.tobytes())
        compressed_x2 = 0
        compressed_x3 = 0

        if debug:
            decompress_x = np.frombuffer(self.dctx.decompress(compressed_x1), dtype=data.dtype)
            assert (np.array_equal(decompress_x.reshape(data.shape), data))
        return sys.getsizeof(compressed_x1), sys.getsizeof(compressed_x2), sys.getsizeof(compressed_x3)

    def snappy_compress(self, x):
        data = x.detach().numpy()
        compressed_x = snappy.compress(data.tobytes())
        if debug:
            decompress_x = np.frombuffer(snappy.decompress(compressed_x), dtype=data.dtype)
            assert (np.array_equal(decompress_x.reshape(data.shape), data))
        return sys.getsizeof(compressed_x)

    def to_sparse(self, x):
        """ converts dense tensor x to sparse format """
        x_typename = torch.typename(x).split('.')[-1]
        sparse_tensortype = getattr(torch.sparse, x_typename)

        indices = torch.nonzero(x)
        if len(indices.shape) == 0:  # if all elements are zeros
            return sparse_tensortype(*x.shape)
        indices = indices.t()
        values = x[tuple(indices[i] for i in range(indices.shape[0]))]
        return sparse_tensortype(indices, values, x.size())

    def rle_compress(self, x):
        # start_time = time.time()
        flatten_x = x.flatten().tolist()
        cur_v = flatten_x[0]
        cur_run = 1
        result = []
        # compress
        for i in range(1, len(flatten_x)):
            v = flatten_x[i]
            if v == cur_v:
                cur_run += 1
            else:
                result.append([cur_v, cur_run])
                cur_v = v
                cur_run = 1
        result.append([cur_v, cur_run])
        trans_x = torch.zeros([1, len(flatten_x)])
        # CalLayer.rle_cpr_time += end_time - start_time
        if debug:
            # decompress
            idx = 0
            for (v, r) in result:
                trans_x[0][idx:idx + r] = v
                idx += r
            trans_x = torch.reshape(trans_x, x.shape)

            if not torch.equal(trans_x, x):
                print(x)
                print("----------")
                print(trans_x)
            assert (torch.equal(trans_x, x))

        return len(result) * 2 * Optimizer.data_size

    def sparse_compress_2D(self, x):
        # start_time = time.time()
        sparse_x = self.to_sparse(x)
        if debug:
            trans_x = sparse_x.to_dense()
            assert (torch.equal(x, trans_x))

        sparse_mem = np.prod(sparse_x._indices().shape) * Optimizer.data_size
        sparse_mem += np.prod(sparse_x._values().shape) * Optimizer.data_size
        # end_time = time.time()
        # CalLayer.sparse_cpr_time += end_time - start_time
        return sparse_mem

    def sparse_compress(self, x, get_min = True):
        sparse_mem = 0
        if len(list(x.shape)) == 4:
            sample_num = x.shape[0]
            filter_num = x.shape[1]
            for img in range(sample_num):
                for f in range(filter_num):
                    cur_sparse_mem = self.sparse_compress_2D(x[img][f])
                    if get_min:
                        sparse_mem += min(cur_sparse_mem, self.org_mem(x[img][f]))
                    else:
                        sparse_mem += cur_sparse_mem

        elif len(list(x.shape)) == 3:
            sample_num = x.shape[0]
            for img in range(sample_num):
                cur_sparse_mem = self.sparse_compress_2D(x[img])
                if get_min:
                    sparse_mem += min(cur_sparse_mem, self.org_mem(x[img]))
                else:
                    sparse_mem += cur_sparse_mem

        elif len(list(x.shape)) == 2:
            cur_sparse_mem = self.sparse_compress_2D(x)
            if get_min:
                sparse_mem += min(cur_sparse_mem, self.org_mem(x))
            else:
                sparse_mem += cur_sparse_mem

        else: # 1D, do nothing for now
            sparse_mem += self.org_mem(x)

        return sparse_mem

if __name__ == "__main__":
    net = mobilenet.MobileNet()
    opt = Optimizer()
    opt.register(net)
    x = torch.randn(1,3,32,32)
    print (net)
    net(x)
