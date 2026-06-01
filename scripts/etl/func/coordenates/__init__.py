import re

__version__ = "0.0.1"

class coordenates:
    def __init__(self, lat:str, lon:str, direction_lat:str = "S",direction_lon:str = "E"):
        self.lat = lat
        self.lon = lon
        self.direction_lat = direction_lat
        self.direction_lon = direction_lon
        pass

    def dms2dd(self) -> list:

        # Separamos os valores de degraus, minutos e segundos

        degrees_lat,minutes_lat,seconds_lat = (re.search(r"(\d+)°", self.lat).group(1),
                                               re.search(r"(\d+)'", self.lat).group(1),
                                               re.search(r"(\d+)\"", self.lat).group(1)
                                               )
        
        degrees_lon,minutes_lon,seconds_lon = (re.search(r"(\d+)°", self.lon).group(1),
                                               re.search(r"(\d+)'", self.lon).group(1),
                                               re.search(r"(\d+)\"", self.lon).group(1)
                                               )
        
        dd_lat, dd_lon = (float(degrees_lat) + float(minutes_lat)/60 + float(seconds_lat)/(60*60),
                          float(degrees_lon) + float(minutes_lon)/60 + float(seconds_lon)/(60*60)
                          )
        if self.direction_lon == 'E' or self.direction_lat == 'S':
            dd_lat *= -1
            dd_lon *= -1

        return dd_lat,dd_lon,3