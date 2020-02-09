import numpy as np
import threading
import math as m

class Sequence(object):
    """Base object for fitting to a sequence of data, such as a dataset. 
    The abstract method `__getitem__` should be overridden and should return a complete batch,
    contained in a tuple.
    The mini-batches are chosen sequentially from an index list that are reshuffled on each epoch.

    Notes:
    `Sequence` are a safer way to do multiprocessing. This structure guarantees
    that the network will only train once
    on each sample per epoch which is not the case with generators.
    """

    def __init__(self, x, y, vectorizer=None, batch_size=32):
        if y is not None and len(x) != len(y):
            raise ValueError('X (features) and y (labels) '
                                'should have the same length. '
                                'Found: X.shape = %s, y.shape = %s' %
                                (np.asarray(x).shape, np.asarray(y).shape))

        self.x = np.asarray(x)

        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None

        self.vectorizer = vectorizer
        self.batch_size = batch_size
        self._shuffle_index()      

    def __getitem__(self, index):
        """Gets batch at position `index`.
        Arguments:
            index: position of the batch in the Sequence.
        Returns:
            A batch
        """
        raise NotImplementedError

    def __len__(self):
        """Number of batch in the Sequence.
        Returns:
            The number of batches in the Sequence.
        """
        l =  m.ceil(len(self.x) / self.batch_size)
        l_last = len(self.x)%l
        if (l_last > 0) and (l_last < self.batch_size/4):
            print("Last batch will be %i samples which is significantly lower than intended %i. This can lead to unstable training."%(l_last, self.batch_size))
        return l

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item    

    def _shuffle_index(self):
        """Method that creates/reshuffles the index for creation 
        of the mini-batches in a random fashion"""
        self.index = np.random.permutation(len(self.x))

    def get_samples_idx(self, idx):
        """Get the indices of the samples, from a given minibatch index"""

        sample_idxs = self.index[idx * self.batch_size:(idx + 1) * self.batch_size]
        return sample_idxs

    def on_epoch_end(self):
        """Method called at the end of every epoch. Reshuffles the index to
        create randomly selected mini-batches.
        """
        self._shuffle_index()

class Iterator(object):
    """Abstract base class for data iterators.

    :parameter n: Integer, total number of samples in the dataset to loop over.
    :parameter batch_size: Integer, size of a batch.
    :parameter shuffle: Boolean, whether to shuffle the data between epochs.
    :parameter seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)
        if n < batch_size:
            raise ValueError('Input data length is shorter than batch_size\nAdjust batch_size')

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)

            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


class SmilesSequence(Sequence):
    """Sequence yielding vectorized SMILES.
    Initialize with X, y and a vectorizer, implementing a batch_wise transform methods"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        """Returns the vectorized samples."""
        samples = self.get_samples_idx(idx)
        batch_x = self.x[samples]
        batch_x = self.vectorizer.transform(batch_x)
        batch_y = self.y[samples]
        return (batch_x, batch_y)

class SmilesGenerator(Iterator):
    """Iterator yielding data from a SMILES array.

    :parameter x: Numpy array of SMILES input data.
    :parameter y: Numpy array of targets data.
    :parameter vectorizer: Instance of molecular vectorizer
    :parameter batch_size: Integer, size of a batch.
    :parameter shuffle: Boolean, whether to shuffle the data between epochs.
    :parameter seed: Random seed for data shuffling.
    :parameter dtype: dtype to use for returned batch. Set to keras.backend.floatx if using Keras
    """

    def __init__(self, x, y, vectorizer,
                 batch_size=32, shuffle=False, seed=None,
                 dtype=np.float32
                 ):
        if y is not None and len(x) != len(y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: X.shape = %s, y.shape = %s' %
                             (np.asarray(x).shape, np.asarray(y).shape))

        self.x = np.asarray(x)

        if y is not None:
            self.y = np.asarray(y)
        else:
            self.y = None
        self.vectorizer = vectorizer
        self.dtype = dtype
        #print(type(self))
        #print(type(SmilesGenerator))
        super(SmilesGenerator, self).__init__(len(x), batch_size, shuffle, seed)

    def next(self):
        """For python 2.x.
        returns the next batch. The X is directly the vectorized format and y is as supplied.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.vectorizer.dims)), dtype=self.dtype)
        for i, j in enumerate(index_array):
            smiles = self.x[j:j+1]
            x = self.vectorizer.transform(smiles)
            batch_x[i] = x

        if self.y is None:
            return batch_x
        batch_y = self.y[index_array]
        return batch_x, batch_y
        
    
class HetSmilesGenerator(SmilesGenerator):
    """Hetero (maybe) generator class, for use to train the autoencoder.
    
    smilesvectorizer creates the input for the encoder
        Can be left_padded
    smilesvectorizer_2 creates the teacher input for the decoder + output.
        Must be right_padded. Output for decoder left shifted 1 pos, so no startchar.
    """
    def __init__(self, x, y, smilesvectorizer, smilesvectorizer_2,
                 batch_size=32, shuffle=False, seed=None,
                 dtype=np.float32):
        super(HetSmilesGenerator,self).__init__(x, y, smilesvectorizer,
                 batch_size=batch_size, shuffle=shuffle, seed=seed,
                 dtype=dtype)
        self.smilesvectorizer = smilesvectorizer
        self.smilesvectorizer_2 = smilesvectorizer_2
        
    def next(self):
        """For python 2.x.

        :returns: The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
            
        self.enc_dims = list(self.smilesvectorizer.dims)
        #Subtract one from the output dims to prepare for the left shifting of output
        self.dec_dims = list(self.smilesvectorizer.dims)
        self.dec_dims[0] = self.dec_dims[0]-1

        #Prepare output arrays
        batch_1D = np.zeros(tuple([current_batch_size] + self.enc_dims), dtype=self.dtype)
        batch_1D_i = np.zeros(tuple([current_batch_size] + self.dec_dims), dtype=self.dtype)
        batch_1D_o = np.zeros(tuple([current_batch_size] + self.dec_dims), dtype=self.dtype)

        #TODO Maybe vectorize this, transform already has a for loop
        for i, j in enumerate(index_array):
            mol = self.x[j:j+1]
            
            chem1d_enc = self.smilesvectorizer.transform(mol)
            chem1d_dec = self.smilesvectorizer_2.transform(mol)
            
            batch_1D[i] = chem1d_enc
            batch_1D_i[i] = chem1d_dec[:,0:-1,:] #Including start_char
            batch_1D_o[i] = chem1d_dec[:,1:,:]  #No start_char

        return [batch_1D, batch_1D_i], batch_1D_o
    