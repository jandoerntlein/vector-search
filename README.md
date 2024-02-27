# Vector Search

Vector-search leverages text representations known as vector embeddings to enable advanced search capabilities. 

## Product Recommendation Engine using Semantic Similarity

This repository contains an example on how to implement [semantic similarity](https://en.wikipedia.org/wiki/Semantic_similarity) search based on vector embeddings. I wrote this small example after reading the articles [here](https://dev.to/stephenc222/introduction-to-vector-search-and-embeddings-4lee) and [here](https://markus.oberlehner.net/blog/your-own-vector-search-in-5-minutes-with-sqlite-openai-embeddings-and-nodejs/). Both previous examples rely on external cloud based services from OpenAI, and I wondered if it was possible to implement this in an offline way. 

The goal of the Python script was to provide an easy entry to the topic and to jump start your own implementation (= proof of concept quality).

So in the following, creation & searching of the database is completely offline, without any external service involved. The same [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) approach is used (= normalized dot product) to measure the distance between vectors. 

In contrast to regular text-based search engines, this method provides a significant performance boost with vector-based search - especially when operating on huge data sets, because costly string comparison is replaced by efficient vector math. 

There is no external cloud infrastructure required, all is based on the embedding model [UAE-Large-V1](https://github.com/SeanLee97/AnglE) by AnglE. This model is in the top 10 of the [Massive Text Embedding Benchmark (MTEB) leaderboard](https://huggingface.co/spaces/mteb/leaderboard) at HuggingFace and provides great results for English text embeddings. Other models could be used in the same way, of course. 

## Prerequisites

Let's get started. Note that I used Python 3.11.7 for testing.

```bash
# Create an isolated python virtual environment first
git clone [this repo] vector-search && cd vector-search
python3 -m venv .
. bin/activate
pip install angle-meb numpy scikit-learn
# or just use 'pip install -r requirements.txt'

# We need git-lfs enabled, to download the 5.6 GB model from HuggingFace (I am on macOS)
brew install git-lfs
git lfs install

# Download the text embedding model
git clone https://huggingface.co/WhereIsAI/UAE-Large-V1
```

Perform a quick test in the REPL to ensure the setup is correct:

```python
from angle_emb import AnglE
angle = AnglE.from_pretrained('./UAE-Large-V1', pooling_strategy='cls').cuda() 
test = angle.encode('test', to_numpy=True)
print(len(test[0])) # quick check: 1024 is the vector length
print(test) # print the vector
```

To run the example:

```python
python3 ./embedding.py
```

## Using SQLite3-VSS as Persistent Data Backend
There is an article [here](https://markus.oberlehner.net/blog/your-own-vector-search-in-5-minutes-with-sqlite-openai-embeddings-and-nodejs/) by Markus Oberlehner, explaining how you could use sqlite-vss as a vector storage database (instead of the in-memory approach presented here). The article also is based on OpenAI cloud access, but can be easily adapted to use UAE-Large-V1 as an offline alternative for the embeddings. sqlite-vss also includes a Vector Search Engine, so you don't have to implement (or choose) an algorithm like cosine_similarity yourself. I did not benchmark it yet, but I would expect it to offer a much better performance as it contains various optimizations. Note that the vector length used in the blog post is 1536, while the approach here has a vector length of 1024, so you need to adjust accordingly if you would like to try it (I did, works as expected).

In addition, to bridge the gap between your NodeJS application and the pythonic database backend, you could expose your API with a [FastAPI](https://fastapi.tiangolo.com) interface (e.g. running in a docker container). A simple API layer could then be available in Javascript. 

## Results & Accuracy
The tests were performed on a database with roughly 100 entries of random products and its descriptions. 

Here you can see an example of the results. The user might be searching for a product like "shirt", "gaming gpu", or "coffee machine" on your shop or search engine, and now gets recommendations of similar products in a side bar. Quite impressive!

```
Since you are looking for "shirt", you also might want to have a look at: 
  - Champion Men's Powerblend Fleece Hoodie - Soft and comfortable hoodie for casual wear.
  - Patagonia Nano Puff Jacket - Lightweight and packable insulated jacket for outdoor activities.
  - L.L.Bean Women's Bean Boots - Iconic duck boots designed for wet weather and outdoor activities.

Since you are looking for "gaming gpu", you also might want to have a look at: 
  - Asus ROG Strix GeForce RTX 3080 - High-end gaming graphics card with ray tracing technology.
  - Sony PlayStation 5 - Next-generation gaming console with powerful performance.
  - Oculus Quest 2 - All-in-one virtual reality headset for immersive gaming.

Since you are looking for "coffee machine", you also might want to have a look at: 
  - Breville Bambino Plus Espresso Machine - Compact espresso machine with automatic milk frothing.
  - Keurig K-Classic Coffee Maker - Single-serve coffee maker with a large water reservoir.
  - KitchenAid Artisan Stand Mixer - Durable mixer for baking and cooking with various attachments.
```

Those results are the outcome of embedding.py, so you can try yourself with different search terms.

## Benchmarks
Preliminary benchmark results from an 2021 M1 MacBook Pro to get a feeling for performance:
- Initializing the model: ~1 sec
- Calculating the embedding vector: between 30 ms and 140 ms
- Searching a product: << 1 ms

With this setup, you can jump start your own vector-search based engine with ease.