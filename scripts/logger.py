import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')

error_handler = logging.FileHandler('./logs/error.log')
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)

info_handler = logging.FileHandler('./logs/info.log')
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(error_handler)
logger.addHandler(info_handler)
logger.addHandler(stream_handler)