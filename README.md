### Neural Style Transfer Web App - [FastAPI + streamlit + Docker]

NST - application based on the Perceptual Losses for Real-Time Style Transfer and Super-Resolution paper and Justin Johnson's pre-trained models.

Application implements:

* Develop an asynchronous API with Python and FastAPI
* Serve up a machine learning model with FastAPI
* Develop a UI with Streamlit
* Containerize FastAPI and Streamlit with Docker
* Leverage asyncio to execute code in the background outside the request/response flow

#### Tools used
- FastAPI: for the API
- streamlit : for the interface
- Docker: to containerize the app

#### Download the models
```
./download_models.sh
```

#### Run
```
docker-compose up -d
```

<div class="col">
<div class="row">
<img  src="https://cs.stanford.edu/people/jcjohns/eccv16/style_results/starry_256_28_mine.jpg" width="300"/> 
<img  src="https://cs.stanford.edu/people/jcjohns/eccv16/style_results/starry_256_26_mine.jpg" width="300"/> 
</div>
</div>

