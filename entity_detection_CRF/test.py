from args import get_args
import os
from model import EntityDetection
args=get_args()
model = EntityDetection(word_vocab, config)
if args.word_vectors:
    if os.path.isfile(args.vector_cache):
        pretrained = torch.load(args.vector_cache)
        model.embed.word_lookup_table.weight.data.copy_(pretrained)
    else:
        pretrained = model.embed.load_pretrained_vectors(args.word_vectors, binary=False,
                                            normalize=args.word_normalize)
        torch.save(pretrained, args.vector_cache)
        print('load pretrained word vectors from %s, pretrained size: %s' %(args.word_vectors,
                                                                                pretrained.size()))