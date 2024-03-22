import math
from datetime import datetime
import whois
from requests import get
from pyquery import PyQuery

class UrlFeaturizer(object):
    """
    UrlFeaturizer class is designed to extract various features from a given URL.
    These features include characteristics of the URL string itself, domain-specific attributes, and page features.
    """
    
    def __init__(self, url):
        """
        Initialize the UrlFeaturizer class.

        Parameters:
        - url (str): The URL of the webpage.
        """
        self.url = url
        self.domain = url.split('//')[-1].split('/')[0]
        self.today = datetime.now().replace(tzinfo=None)

        try:
            self.whois = whois.query(self.domain).__dict__
        except:
            self.whois = None

        try:
            self.response = get(self.url)
            self.pq = PyQuery(self.response.text)
        except:
            self.response = None
            self.pq = None

    ## URL string Features
    def entropy(self):
        """
        Calculate the entropy of the URL string.
        """
        string = self.url.strip()
        prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
        entropy = sum([(p * math.log(p) / math.log(2.0)) for p in prob])
        return entropy

    def ip(self):
        """
        Check if the URL contains an IP address.
        """
        string = self.url
        flag = False
        if ("." in string):
            elements_array = string.strip().split(".")
            if(len(elements_array) == 4):
                for i in elements_array:
                    if (i.isnumeric() and int(i)>=0 and int(i)<=255):
                        flag=True
                    else:
                        flag=False
                        break
        if flag:
            return 1 
        else:
            return 0

    def numDigits(self):
        """
        Count the number of digits in the URL
        """
        
        digits = [i for i in self.url if i.isdigit()]
        return len(digits)

    def urlLength(self):
        """
        Calculate the length of the URL
        """
        return len(self.url)

    def numParameters(self):
        """
        Count the number of parameters in the URL.
        """
        params = self.url.split('&')
        return len(params) - 1

    def numFragments(self):
        """
        Count the number of fragments in the URL
        """
        fragments = self.url.split('#')
        return len(fragments) - 1

    def numSubDomains(self):
        """
        Count the number of subdomains in the URL
        """
        subdomains = self.url.split('http')[-1].split('//')[-1].split('/')
        return len(subdomains)-1

    def domainExtension(self):
        """
        Extract the extension of the domain from the URL.

        Returns:
        - ext (str): Domain extension extracted from the URL.
        """
        ext = self.url.split('.')[-1].split('/')[0]
        return ext

    ## URL domain features
    def hasHttp(self):
        """
        Check if the URL contains 'http:'.

        Returns:
        - bool: True if 'http:' is present in the URL, False otherwise.
        """
        return 'http:' in self.url

    def hasHttps(self):
        """
        Check if the URL contains 'https:'.

        Returns:
        - bool: True if 'https:' is present in the URL, False otherwise.
        """
        return 'https:' in self.url

        # Define a method to extract days since registration
    def daysSinceRegistration(self):
        """
        Calculate the number of days since the domain was registered.

        Returns:
        - float: Number of days since domain registration.
        """
        if self.whois and self.whois['creation_date']:
            diff = self.today - self.whois['creation_date'].replace(tzinfo=None)
            diff = diff.total_seconds() / (60 * 60 * 24)  # Convert to float representing number of days
            return diff
        else:
            return 0

    # Define a method to extract days since expiration
    def daysSinceExpiration(self):
        """
        Calculate the number of days until the domain's expiration date.

        Returns:
        - float: Number of days until domain expiration.
        """
        if self.whois and self.whois['expiration_date']:
            diff = self.whois['expiration_date'].replace(tzinfo=None) - self.today
            diff = diff.total_seconds() / (60 * 60 * 24)  # Convert to float representing number of days
            return diff
        else:
            return 0
        
    ## URL Page Features
    def bodyLength(self):
        """
        Calculate the length of the text content in the HTML body of the webpage.

        Returns:
        - int: Length of the text content in the HTML body.
        """
        if self.pq is not None:
            return len(self.pq('html').text()) if self.urlIsLive else 0
        else:
            return 0

    def numTitles(self):
        """
        Count the number of title tags (h1 to h6) in the HTML of the webpage.

        Returns:
        - int: Number of title tags found in the HTML.
        """
        if self.pq is not None:
            titles = ['h{}'.format(i) for i in range(7)]
            titles = [self.pq(i).items() for i in titles]
            return len([item for s in titles for item in s])
        else:
            return 0

    def numImages(self):
        """
        Count the number of image tags in the HTML of the webpage.

        Returns:
        - int: Number of image tags found in the HTML.
        """
        if self.pq is not None:
            return len([i for i in self.pq('img').items()])
        else:
            return 0

    def numLinks(self):
        """
        Count the number of anchor tags (links) in the HTML of the webpage.

        Returns:
        - int: Number of anchor tags found in the HTML.
        """
        if self.pq is not None:
            return len([i for i in self.pq('a').items()])
        else:
            return 0

    def scriptLength(self):
        """
        Calculate the length of the JavaScript code present in the webpage.

        Returns:
        - int: Length of the JavaScript code in the webpage.
        """
        if self.pq is not None:
            return len(self.pq('script').text())
        else:
            return 0

    def specialCharacters(self):
        """
        Count the number of special characters in the HTML body of the webpage.

        Returns:
        - int: Number of special characters found in the HTML body.
        """
        if self.pq is not None:
            bodyText = self.pq('html').text()
            schars = [i for i in bodyText if not i.isdigit() and not i.isalpha()]
            return len(schars)
        else:
            return 0

    def scriptToSpecialCharsRatio(self):
        """
        Calculate the ratio of script length to the number of special characters in the HTML body.

        Returns:
        - float: Ratio of script length to special characters.
        """
        v = self.specialCharacters()
        if self.pq is not None and v!=0:
            sscr = self.scriptLength()/v
        else:
            sscr = 0
        return sscr

    def scriptTobodyRatio(self):
        """
        Calculate the ratio of script length to body length.

        Returns:
        - float: The ratio of script length to body length.
        """
        v = self.bodyLength()
        if self.pq is not None and v != 0:
            sbr = self.scriptLength() / v
        else:
            sbr = 0
        return sbr

    def bodyToSpecialCharRatio(self):
        """
        Calculate the ratio of special characters in the body of the webpage.

        Returns:
        - float: The ratio of special characters to body length.
        """
        v = self.bodyLength()
        if self.pq is not None and v != 0:
            bscr = self.specialCharacters() / v
        else:
            bscr = 0
        return bscr

    def urlIsLive(self):
        """
        Check if the URL is live (returns 200).

        Returns:
        - bool: True if the URL is live (returns 200), False otherwise.
        """
        return self.response == 200
    
    # Define a method to extract features
    def extract_features(self):
        """
        Extract features from the URL.

        Returns:
        - list: A list containing the values of all the features extracted from the URL.
        """
        features = [
            self.entropy(),
            self.ip(),
            self.numDigits(),
            self.urlLength(),
            self.numParameters(),
            self.numFragments(),
            self.numSubDomains(),
            self.hasHttp(),
            self.hasHttps(),
            self.daysSinceRegistration(),
            self.daysSinceExpiration(),
            self.bodyLength(),
            self.numTitles(),
            self.numImages(),
            self.numLinks(),
            self.scriptLength(),
            self.specialCharacters(),
            self.scriptToSpecialCharsRatio(),
            self.scriptTobodyRatio(),
            self.bodyToSpecialCharRatio(),
            self.urlIsLive()
        ]
        return features