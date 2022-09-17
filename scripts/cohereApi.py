import cohere
co = cohere.Client('{apiKey}')
response = co.generate(
  model='large',
  prompt='-- Given a product and keywords, this program will generate exciting product descriptions. Here are some examples:---\n\nProduct: Monitor\nKeywords: curved, gaming\nExciting Product Description: When it comes to serious gaming, every moment counts. This curved gaming monitor delivers the unprecedented immersion you need to play your best.\n--\nProduct: Surfboard\nKeywords: 6”, matte finish\nExciting Product Description: This 6” surfboard is designed for fun. It\'s a board that almost anyone can pick up, ride, and get psyched on.\n--\nProduct: Headphones\nKeywords: bluetooth, lightweight\nExciting Product Description:\n--\nProduct: \nKeywords: \nExciting Product Description:\n--',
  max_tokens=50,
  temperature=0.8,
  k=0,
  p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop_sequences=["--"],
  return_likelihoods='NONE')
print('Prediction: {}'.format(response.generations[0].text))