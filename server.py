import torch
from torch.autograd import Variable

from sanic import Sanic
from sanic.response import json

import embeddings

from cove import MTLSTM


app = Sanic()
glove = embeddings.GloveEmbedding('common_crawl_840', d_emb=300, show_progress=True, default='zero')
kazuma = embeddings.KazumaCharEmbedding(show_progress=True)


def predict(words):
    m = app.config['model']
    glove_emb = Variable(torch.Tensor([glove.emb(w) for w in words])).unsqueeze(0)
    cove_emb = m(glove_emb, lengths=torch.LongTensor([len(words)]))
    kazuma_emb = Variable(torch.Tensor([kazuma.emb(w) for w in words])).unsqueeze(0)
    emb = torch.cat([cove_emb, glove_emb, kazuma_emb], 2).squeeze(0)

    return emb


@app.route("/", methods=['POST'])
async def test(request):
    d = request.json
    r = {'error': None}
    if 'words' not in d:
        r['error'] = 'Missing field "words"'
    else:
        r['answer'] = predict(d['words']).data.tolist()
    return json(r)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=-1)
    args = parser.parse_args()
    print(args)
    app.config['model'] = MTLSTM()
    app.config['model'].eval()
    if args.gpu > -1:
        assert args.workers == 1, 'Cannot use CUDA when number of workers exceeds 1'
        app.config['model'].cuda(args.gpu)
    print('starting server')
    app.run(host=args.host, port=args.port, workers=args.workers)
