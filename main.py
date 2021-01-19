from telegram_bot import telegram_bot

if __name__ == '__main__':
    from my_bot_key import botkey
    bot = telegram_bot(botkey, model_path="model_files/yolov4_coco.weights", names_path="model_files/coco.names")
    bot.start()