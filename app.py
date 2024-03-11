from fastapi import FastAPI
import uvicorn
import os
from utils.logging import Logger_Days
from config_app.config import get_config
from main import get_result_predict
import json
import numpy as np

config_app = get_config()

if not os.path.exists("./logs"):
    os.makedirs("./logs")
file_name = './logs/logs'
log_obj = Logger_Days(file_name)

app = FastAPI()

numberrequest = 0
@app.get('/vhkt_wo_predict')
async def vhkt_wo_predict():
    global numberrequest
    numberrequest = numberrequest + 1
    print("numberrequest", numberrequest)
    log_obj.info("-------------------------NEW_SESSION_PREDICT----------------------------------")
    log_obj.info("NumberRequest: " +str(numberrequest))
    result = get_result_predict()
    log_obj.info("result: " +str(result))
    return result

# @app.post('/vhkt_wo_training')
# async def post(InputText: str = Form(...), IdRequest: str = Form(...), NameBot: str = Form(...), User: str = Form(...), request: Request = None):
#     global numberrequest
#     numberrequest = numberrequest + 1
#     print("numberrequest", numberrequest)
#     log_obj.info("-------------------------NEW_SESSION----------------------------------")
#     log_obj.info("IP_Client: " +str(request.client.host))
#     log_obj.info("NumberRequest: " +str(numberrequest))
#     result = predict_llm(InputText, IdRequest, NameBot, User, log_obj)
#     return result


uvicorn.run(app, host=config_app['server']['ip_address'], port=int(config_app['server']['port']))