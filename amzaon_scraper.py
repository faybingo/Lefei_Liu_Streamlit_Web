#Use BeautifulSoup to fetch Amazon data is hard, 
# so I've searched through many links to utilize "autoscraper". 
# Scraping data by using this package is much more convenient.
from autoscraper import AutoScraper

#Ariana Grande's amazon data
url_ag='https://www.amazon.com/s?k=ariana+grande+album&crid=890H5WCN4AQF&sprefix=ariana+grande+album%2Caps%2C130&ref=nb_sb_noss_1'

wanted_list = [
               "100+ bought in past month",
                 '$27.98']
scraper = AutoScraper()
result_ag = scraper.build(url_ag, wanted_list)
data_ag=scraper.get_result_similar(url_ag,grouped=True)
print(data_ag)

#this is the code of store csv data_ag
#df_ag = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in data_ag.items()]))
#df_ag.to_csv('amazon_ag.csv', index=False)
#print("Data saved to 'amazon_ag.csv'")




#Post Malone's amazon data
url_pm='https://www.amazon.com/s?k=post+malone+album&crid=3VWSXSPAT530H&sprefix=Post+Malone+al%2Caps%2C172&ref=nb_sb_ss_ts-doa-p_2_14'

wanted_list = [
               "100+ bought in past month",
                 '$39.98']
scraper = AutoScraper()
result_pm = scraper.build(url_pm, wanted_list)
data_pm=scraper.get_result_similar(url_pm,grouped=True)
print(data_pm)

#this is the code of store csv data_pm
# df_ag = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in data_pm.items()]))
# df_ag.to_csv('amazon_pm.csv', index=False)
# print("Data saved to 'amazon_pm.csv'")



#It seems like Daddy Yankee is an old star, 
#so the sales amount per month of his album is less than 50. 
#His amazon page does not display data below 50. It is pity...

# #Daddy Yankee's amazon data
# url_pm='https://www.amazon.com/s?k=daddy+yankee+album&crid=V4B7ERI477JN&qid=1713167710&sprefix=daddy+yankee+album%2Caps%2C304&ref=sr_pg_1'

# wanted_list = [
#                "",
#                  '$39.98']
# scraper = AutoScraper()
# result_dy = scraper.build(url_dy, wanted_list)
# data_dy=scraper.get_result_similar(url_dy,grouped=True)
# print(data_dy)
# df_dy = pd.DataFrame(dict([(k, pd.Series(v)) for k,v in data_dy.items()]))
# df_dy.to_csv('amazon_dy.csv', index=False)
# print("Data saved to 'amazon_dy.csv'")




