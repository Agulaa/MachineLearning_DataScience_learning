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
    response = respond(message)
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
