from perspective import PerspectiveAPI
p = PerspectiveAPI("AIzaSyArDzdCdKuLJB5EjD6YZRwAbc6QebID7TU")
result = p.score("you suck")

print("Toxicity score: " + str(result["TOXICITY"]))