import numpy as np

class DataWrapper:
    def __init__(self, data):
        # Get the size of the data
        self.size = len(data)
        self.x_title = []
        self.x_body = []
        self.y = []
        self.seqlen_title = []
        self.seqlen_body = []
        self.current_batch = 0

        for sample in data:
            self.x_title.append(sample[0])
            self.x_body.append(sample[1])
            self.y.append(sample[2])
            self.seqlen_title.append(len(sample[0]))
            self.seqlen_body.append(len(sample[1]))

        max_seqlen = max(self.seqlen_body + self.seqlen_title)

        #padding the samples with zero vectors
        for i in range(len(self.x_title)):
            self.x_title[i] += [[0]*50] * (max_seqlen - len(self.x_title[i]))
            self.x_body[i] += [[0]*50] * (max_seqlen - len(self.x_body[i]))

    # return the next batch of the data from the data set.
    def next(self,batch_size):
        if self.current_batch + batch_size<self.size:
            self.current_batch += batch_size
            return self.x_title[self.current_batch:self.current_batch + batch_size], self.x_body[self.current_batch:self.current_batch + batch_size], self.y[self.current_batch:self.current_batch + batch_size], self.seqlen_title[self.current_batch:self.current_batch+batch_size], self.seqlen_body[self.current_batch:self.current_batch + batch_size]
        else:
            temp = self.current_batch
            self.current_batch = self.current_batch+batch_size-self.size
            batch_x_title = self.x_title[temp:]+self.x_title[:self.current_batch]
            batch_x_body = self.x_body[temp:]+self.x_body[:self.current_batch]
            batch_y = self.y[temp:] + self.y[:self.current_batch]
            batch_seqlen_title = self.seqlen_title[temp:] + self.seqlen_title[:self.current_batch]
            batch_seqlen_body = self.seqlen_body[temp:] + self.seqlen_body[:self.current_batch]
            return batch_x_title,batch_x_body,batch_y,batch_seqlen_title,batch_seqlen_body

    # return the length of the longest sequence
    def max_seqlen(self):
        return max(self.seqlen_body + self.seqlen_title)
