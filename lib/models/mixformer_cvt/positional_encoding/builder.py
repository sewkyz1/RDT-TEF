def build_position_embedding(z_shape, x_shape, dim):
        from .sine import SinePositionEmbedding

        with_branch_index = True

        return SinePositionEmbedding(dim, (z_shape[1], z_shape[0]), 0 if with_branch_index else None), \
               SinePositionEmbedding(dim, (x_shape[1], x_shape[0]), 1 if with_branch_index else None)

