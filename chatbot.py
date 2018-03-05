import random
import re


bot_template = "BOT : {0}"
user_template = "USER : {0}"
question = {
    "what's your name?" : ['my name is EchoBot','they call me EchoBot', "the name's Bot, Echo Bot"],
    "what's the weather today?": ["it's sunny!"]
}
sentence = {
    'I like you.':['Thank you']
}
responses_all = {
    'question': question, 'sentence':sentence
}
responses = {
    "what's your name?" : ['my name is EchoBot', 'they call me EchoBot', "the name's Bot, Echo Bot"],
    "what's the weather today?": ["it's sunny!"], 'I like you.': ['Thank you']
}

keywords = {
    'greet': ['hello', 'hi', 'hey'],
    'thankyou': ['thank', 'thx'],
    'goodbay': ['bye', 'farewell']
}
key_responses = {
    'default': 'default message',
    'goodbye': 'goodbye for now',
    'greet': 'Hello you! :)',
    'thankyou': 'you are very welcome'
}


# Define a dictionary of patterns
patterns = {}
def define_dictionary():

    # Iterate over the keywords dictionary
    for intent, keys in keywords.items():
        # Create regular expressions and compile them into pattern objects
        patterns[intent] = re.compile('|'.join(keys))
    return patterns


patterns = define_dictionary()


# Define a function to find the intent of a message
def match_intent(message):
    matched_intent = None

    for intent, pattern in patterns.items():
        # Check if the pattern occurs in the message
        if pattern.search(message):
            matched_intent = intent
    return matched_intent


# Define a respond with key function
def respond_key(message):
    # Call the match_intent function
    intent = match_intent(message)
    # Fall back to the default response
    key = "default"
    if intent in key_responses:
        key = intent
    return key_responses[key]


def send_message_to_bot():
    message = input('What do you want to send to bot ?')
    return message


# Define a function that responds to a user's message: respond
def respond(message):
    if message in responses:
        if message.endswith('?'):
            return random.choice(responses_all['question'][message])
        else:
            return random.choice(responses_all['sentence'][message])
    else:
        # Concatenate the user's message to the end of a standard bot respone
        bot_message = "I can hear you! You said: " + message
        # Return the result
        return bot_message


# Define a function that sends a message to the bot: send_message
def send_message(message):
    # Print user_template including the user_message
    print(user_template.format(message))
    # Get the bot's response to the message
    response = respond_key(message)
    # Print the bot template including the bot's response.
    print(bot_template.format(response))


# Define replace_pronouns()
def replace_pronouns(message):

    message = message.lower()
    if 'me' in message:
        # Replace 'me' with 'you'
        return re.sub('me', 'you', message)
    if 'my' in message:
        # Replace 'my' with 'your'
        return re.sub('my', 'your', message)
    if 'your' in message:
        # Replace 'your' with 'my'
        return re.sub('your', 'my', message)
    if 'you' in message:
        # Replace 'you' with 'me'
        return re.sub('you', 'me', message)

    return message


# Define find_name()
def find_name(message):
    name = None
    # Create a pattern for checking if the keywords occur
    name_keyword = re.compile(r"\b(name|call)\b")
    # Create a pattern for finding capitalized words
    name_pattern = re.compile('[A-Z]{1}[a-z]*')
    if name_keyword.search(message):
        # Get the matching words in the string
        name_words = name_pattern.findall(message)
        if len(name_words) > 0:
            # Return the name if the keywords are present
            name = ' '.join(name_words)
    return name


# Define respond()
def respond(message):
    # Find the name
    name = find_name(message)
    if name is None:
        return "Hi there!"
    else:
        return "Hello, {0}!".format(name)


def check_replace_pro():
    print(replace_pronouns("my last birthday"))
    print(replace_pronouns("when you went to Florida"))
    print(replace_pronouns("I had my own castle"))


def main():
    # Send a message to the bot

    resp = True
    while resp:
        message = send_message_to_bot()
        send_message(message)
        if message == 'Nara':
            resp = False


if __name__=='__main__':
    main()
