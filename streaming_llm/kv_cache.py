import torch


def slice2d(x, start, end):
    return x[:, :, start:end, ...]


def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]


def slice1d(x, start, end):
    return x[:, start:end, ...]


DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}


class StartRecentKVCache:
    def __init__(
        self,
        start_size=4,
        recent_size=255,
        middle_size=255,
        k_seq_dim=2,
        v_seq_dim=2,
        importance_filtering=False
    ):
        print(f"StartRecentKVCache: {start_size}, {recent_size}, {middle_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.middle_size = middle_size
        #self.cache_size = start_size + recent_size + middle_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        self.importance_weights = None
        if importance_filtering:
            self.cache_size = start_size + recent_size + middle_size
            self.importance_weights = torch.tensor([])


    def __call__(self, past_key_values, importance):
        if past_key_values is None:
            return None
        # past_key_values: ((1, 32, seq_len, 128), (1, 32, seq_len, 128))
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if self.importance_weights is not None:
            if seq_len <= self.cache_size:
                self.importance_weights = torch.cat((self.importance_weights, importance), dim=-1)
                return past_key_values
            else:
                #print(f'seq_len: {seq_len}')
                #print(f'self.importance_weights: {len(self.importance_weights)}')
                #print(f'importance: {len(importance)}')
                assert len(self.importance_weights) + len(importance) > self.cache_size
                self.importance_weights = torch.cat((self.importance_weights, importance), dim=-1)
                candidates = self.importance_weights[self.start_size:-self.recent_size]
                indices = torch.argsort(candidates, dim=-1, descending=True)[:self.middle_size]
                weight_indices = indices + self.start_size
                self.importance_weights = torch.cat([self.importance_weights[:self.start_size],\
                    self.importance_weights[weight_indices], self.importance_weights[-self.recent_size:]], dim=-1)
                #print(self.importance_weights.shape)
                return [
                [
                    # Select and update according to importance weights
                    torch.cat(
                        [
                            self.k_slice(k, 0, self.start_size),
                            k[:, :, weight_indices, ...],
                            self.k_slice(k, seq_len - self.recent_size, seq_len),
                        ],
                        dim=self.k_seq_dim,
                    ),
                    torch.cat(
                        [
                            self.v_slice(v, 0, self.start_size),
                            v[:, :, weight_indices, ...],
                            self.v_slice(v, seq_len - self.recent_size, seq_len),
                        ],
                        dim=self.v_seq_dim,
                    ),
                ]
                for k, v in past_key_values
                ]
            
        if seq_len <= self.cache_size:
                return past_key_values
        return [
            [
                # Select two blocks of indices
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def update(self, past_key_values):

        return
    

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        #print(len(past_key_values))
        #print(f'k: {past_key_values[0][0].shape}')
        #print(f'v: {past_key_values[0][1].shape}')
        # past_key_values: (32, tuple((1, 32, seq_len, 128), (1, 32, seq_len, 128))) 
        # first '32' means 32 transformer blocks
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        
        
        new_past_key_values = [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size), # always the same!
                        self.k_slice(
                            k, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size), # always the same!
                        self.v_slice(
                            v, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]
        #print(f'new_seq_len: {new_past_key_values[0][0].size(self.k_seq_dim)}')
        return new_past_key_values

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, start),
                        self.k_slice(k, end, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start),
                        self.v_slice(v, end, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]
