# coding: utf-8


from datetime import datetime
import numpy as np
from ...components.streamer import ImageStreamerElecom, StreamerTemplate
from ...io import S3ImageIO, IOTemplate


class DataCollector:
    def __init__(self, streamer: StreamerTemplate, io: IOTemplate) -> None:
        if not isinstance(streamer, StreamerTemplate):
            raise TypeError(f"{type(streamer)} is not implemented.")
        if not isinstance(io, IOTemplate):
            raise TypeError(f"{type(io)} is not implemented.")
        self.io = io
        self.streamer = streamer
        
    def run(self) -> str:
        """Retrieve an image from a camera and save the image to a bucket with filename 
        is datetime.

        Returns:
            str: filename.
        """

        image = self.streamer.capture()
        file_name = self.__create_file_name()
        print("save file: ", file_name)
        self.io.save(image, file_name)

        return file_name

    def save(self, image: np.ndarray) -> str:
        """save the image to a bucket with filename is datetime.

        Args:
            image: np.ndarray: input image.

        Returns:
            str: filename.
        """

        file_name = self.__create_file_name()
        print("save file: ", file_name)
        self.io.save(image, file_name)

        return file_name

    def __create_file_name(self) -> str:
        """ creates a image file name with png. 
        The format of the file name is, year, month, day, hour, minutes, and second.

        Returns:
            str: file name
        """
        
        now = datetime.now()
        date_time = now.strftime("%Y%m%d%H%M%S")
        file_name = date_time + ".png"

        return file_name




