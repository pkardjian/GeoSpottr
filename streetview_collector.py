import requests
import os
import json
from random import randint

url = 'https://maps.googleapis.com/maps/api/streetview'
cities = {}

def load_cities(input_directory):
    for filename in os.listdir(input_directory):
        
        path = os.path.join(input_directory, filename)
        print(path)
        file_name = os.path.basename(path)
        city_name = file_name.split('.')[0]
        file_extension = file_name.split('.')[1]
        if file_extension != 'geojson':
            continue
        
        with open(path) as f:
          coordinates = []
          print(f'Loading addresses...')
          for line in f:
              data = json.loads(line, strict=False)
              if (data is None or data['geometry'] is None):
                continue
              coordinates.append(data['geometry']['coordinates'])
          cities[city_name] = coordinates

    return cities

def main(input_directory):
    output_directory = '/Users/pascalkardjian/Downloads/output' # WHERE YOU WANT TO STORE IMAGES
    os.mkdir(output_directory) 
    
    cities = load_cities(input_directory)
    
    for city, coords in cities.items():
      print(city)
      os.mkdir(output_directory + f'/{city}')
      for j in range(1,1251):
          addressLoc = coords[randint(0, len(coords) - 1)]
          coords.remove(addressLoc) # remove so we don't get the same coordinates twice

          # Set the parameters for the API call to Google Street View
          params = {
              'key': 'AIzaSyCJ0zPPP5ZV-rS96TAFPypwNqCFMtkWUzw',
              'size': '640x640',
              'location': str(addressLoc[1]) + ',' + str(addressLoc[0]),
              'heading': str((randint(0, 3) * 90) + randint(-15, 15)),
              'pitch': '20',
              'fov': '90'
              }
          
          response = requests.get(url, params)
          
          with open(os.path.join(output_directory + f'/{city}', f'street_view_{j}.jpg'), "wb") as file:
              file.write(response.content)

directory = '/Users/pascalkardjian/Downloads/temp' # WHERE ADDRESS BOOK IS STORED
main(directory)
