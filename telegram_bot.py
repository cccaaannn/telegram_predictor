from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import requests
import logging
import shutil
import uuid
import cv2
import os

from yolo_predictor import yolo_predictor
from yolo_drawer import yolo_drawer

class telegram_bot():
    def __init__(self,
        botkey,
        model_path, 
        names_path,
        downloaded_image_save_folder="images/downloaded",
        predicted_image_save_folder="images/predicted",
        logger_file_name="logs/telegram_bot.log"
        ):

        self.__botkey = botkey
        self.model_path = model_path
        self.names_path = names_path 
        self.downloaded_image_save_folder = downloaded_image_save_folder
        self.predicted_image_save_folder = predicted_image_save_folder

        # start logging 
        logging.basicConfig(level=logging.INFO, format="[Telegram predictor] [%(levelname)s] (%(asctime)s) %(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(), logging.FileHandler(filename=logger_file_name)])

        self.__start_predictor()

    def __start_predictor(self):
        logging.info("Model is loading")
        self.__predictor = yolo_predictor(model_path=self.model_path, names_path=self.names_path)
        self.__drawer = yolo_drawer()

    def __download_file_requests(self, url, local_full_path):
        with requests.get(url, stream=True) as req:
            with open(local_full_path, 'wb') as file:
                shutil.copyfileobj(req.raw, file)

    def __download_image(self, file_id):
        """uses telegrams ap to retrive image"""
        
        # get file path on server
        file_path_api_str = "https://api.telegram.org/bot{0}/getFile?file_id={1}".format(self.__botkey, file_id)
        response = requests.get(file_path_api_str).json()
        file_path_on_server = response["result"]["file_path"]
        _ , file_name_on_server = os.path.split(file_path_on_server)
        
        file_download_api_str = "https://api.telegram.org/file/bot{0}/{1}".format(self.__botkey, file_path_on_server)

        # create unique file name on that path to prevent override
        unique_filename = str(uuid.uuid4())
        _, file_extension = os.path.splitext(file_name_on_server)
        unique_full_local_path = os.path.join(self.downloaded_image_save_folder, unique_filename + file_extension)

        self.__download_file_requests(file_download_api_str, unique_full_local_path)

        return unique_full_local_path


    def start(self):
        def error(update, context):
            logging.warning('Update {0} caused error {1}'.format(update, context.error))

        def help(update, context):
            update.message.reply_text("Send an image for predicting")

        def prediction_handler(update, context):
            try:    
                # download image from api
                file_id = update.message.photo[-1].file_id
                image_path = self.__download_image(file_id)
                update.message.reply_text("Predicting...")

                # predict
                logging.info("Predicting image:{0}".format(image_path))
                predictions = self.__predictor.predict(image_path)

                if(predictions):
                    str_predictions = ""
                    for pred in predictions:
                        str_predictions += "{0}  %{1:.2f}\n".format(pred[0], pred[2])

                    update.message.reply_text(str_predictions)

                    # draw and sed labeld image
                    _, save_path = self.__drawer.draw(predictions, image_path, show=False, save_folder_path=self.predicted_image_save_folder, resize=None, saved_file_suffix="")
                    context.bot.send_photo(chat_id=update.message.chat.id, photo=open(save_path, 'rb'))

                    logging.info("Predicted image:{0} result:{1}".format(save_path, predictions))
                else:
                    logging.info("Nothing detected image:{0}".format(image_path))
                    update.message.reply_text("Nothing detected")

            except Exception:
                logging.exception("", exc_info=True)
                update.message.reply_text("oops something went wrong")


        logging.info("Bot starting")
        updater = Updater(self.__botkey, use_context=True)

        updater.dispatcher.add_error_handler(error)
        updater.dispatcher.add_handler(CommandHandler("help", help))
        updater.dispatcher.add_handler(MessageHandler(Filters.photo, prediction_handler))

        updater.start_polling()
        updater.idle()

