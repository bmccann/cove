# Start from base python/pytorch container
FROM bmccann/py36-torch02

ADD ./ /cove/
RUN cd cove && pip install -r requirements.txt && python setup.py develop

VOLUME /cove/cove/.torch
VOLUME /cove/.embeddings

EXPOSE 8000
ENV GPU=-1
CMD python /cove/server.py --gpu $GPU
