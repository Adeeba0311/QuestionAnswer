import wikipedia as wp

def fetch_wikipedia_page(query):
    res = wp.search(query, results = 5)
    # print(res)
    # result = wikipedia.summary(res[0])
    # print(result)
    res_page = wp.page(res[0]).content
    #print(res_page)
    return res_page
    
query = "www"
print(fetch_wikipedia_page(query))
