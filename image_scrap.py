from bing_image_downloader import downloader
#downloader.download("Steve Jobs", limit=250, output_dir="dataset/Steve Jobs")

Names = ["Johnny Depp", "John Travolta", "Leonardo Di Caprio", "Brad Pitt", "Keanu Reeves",
         "Kristen Stewart", "Angelina Jolie", "Natalie Portman","Nicole Kidman", "Penelope Cruz"]


for element in Names:
  downloader.download(element, limit=200, output_dir="dataset/")