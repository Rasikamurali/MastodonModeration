import pandas as pd 
import numpy as np 
import os 
import requests
from matplotlib import pyplot as plt 
from sklearn.manifold import TSNE
from transformers import BertTokenizer, BertModel
import torch
import mplcursors

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


rules_data = rules_data = [
    ["Don't be a dick."],
    ['Private server for Tian'],
    ['> be Lily'], 
    ['> be so hot that everyone who sees you instantly falls in love'], 
    ['> be so confident that it radiates from you'], 
    ['> be the kind of girl who knows what she wants and takes it'], 
    ['> be the definition of sexy'], 
    ['> be Lily Velour'],
    ['No content illegal in The United Kingdom'], 
    ['No incitement of violence or promotion of discriminatory or violent ideologies'], 
    ['No spam, harassment, dogpiling or doxxing of other users, including mass spam or mass reporting'], 
    ['No intentionally false or misleading information. By extension, do not pose as persons of interest.'], 
    ['Racism, Sexism, Homophobia, transphobia, xenophobia and other forms of hate and discrimination are not welcome'], 
    ['Sexually explicit or violent media must be marked as sensitive when posting'],
    ['You must be Jon Nemargut to use this Mastodon instance. '],
    ["Don't get banned."],
    ["Never Underestimate Yourself, or Anyone Else. All Y'ALL.\r\n", 
    'When at all frustrated by anything, take it outside. (and go for a hike).\r\n\r\n'], 
    ['Be Kind. Kind can be Nice, but is not the same. We are Kind here.'],
    ['El contenido sexual explícito o violento no está permitido.', 
    'No está permitido el racismo, clasismo, sexismo, homofobia, transfobia, xenofobia o cualquier otra actitud de discriminación.', 
    'No se permite incitación a la violencia ni promoción de ideologías violentas.'],
    ['No harassment, organized or disorganized. That is, one person harassing someone or a group of people harassing someone will be treated the same.'], 
    ['No doxxing'], 
    ['No nazis or related propaganda'], 
    ['Use CW (content warning) setting for NSFW material and mark as sensitive'], 
    ['No hate, racism, sexism, homophobia, transphobia, xenophobia, or casteism'], 
    ['No incitement of violence or promotion of violent ideologies'],
    ['By registering or login here, you must read and agree to our ToS, Privacy Policy and Code of Ethics. You acknowledge that your IP & email addresses can be logged for security purposes. Skip signin / signup if you do not agree.', 
    'You can provide interesting links and commentary on longer articles / content published elsewhere. If you just send in a little blurb saying “I like xyz.com app and think they are great” it goes to junk.', 
    'Though if statements are just a kind of multiverse geolocation device being embedded into the connected mob as standalone, drive with care! You can send detailed suggestions, grievances or complaints to editor@wisepoint.org with your full profile, phone and email.'],
    ['No racism, sexism, homophobia, transphobia, xenophobia, or casteism'], 
    ['No incitement or depictions of violence, or the promotion of violent ideologies'], 
    ['No harassment, dogpiling or doxxing of other users'], 
    ['NSFW content (nudity, sexual acts) is permitted so long as it\'s marked as sensitive AND behind a content warning'], 
    ['No illegal content (depictions of minors, drug use, etc)'],
    ['Do not share, trade, link to, or distribute in any manner here, any copyright material, without the express written consent of the owner.'], 
    ['No Prohibited Content: It is prohibited to link to pornography or content that can damage computers or data such as viruses or malware.'], 
    ['Spam is not tolerated on BoLS.'], 
    ['Do not use racist slurs, prejudiced comments, profanity, derogatory statements, or hate speech. Evading the moderation filters by using punctuation marks or purposefully misspelling words is considered an offense.'], 
    ['Do not initiate arguments of a personal nature. Be courteous to other users, even if you disagree with someone or believe that they are ill-informed. Refrain from flaming posts or comments. You may argue and debate, but do not make any attacks of a personal nature (including grammar trolling).'],
    ['Only photos of T-shirts or designs'],
    ['Sexually explicit or violent media must be marked as sensitive when posting'], 
    ['No discrimination, including racism, sexism, homophobia, transphobia, xenophobia, or casteism'], 
    ['No incitement of violence or promotion of violent ideologies'], 
    ['No harassment, dogpiling or doxxing of other users'], 
    ['Do not share false or misleading information that may lead to physical harm'], 
    ['No accounts owned by Elon Musk.'],
    ['Jeder wird respektvoll behandelt, egal welchen Glaubens, Herkunft, sexuelle Identität oder Ausrichtung. Höflicher Umgang wird bevorzugt. #NoAfd. Bei Nichtbeachtung, Ausschluss!'],
    ['Be nice'], 
    ['Nothing hateful, toxic, or illegal'],
    ['This server is for the members of the Van Gogh-Goghs Comedy Collective (https://vgg.com).'], 
    ['The server is not open to the general public. But if you are or know one of the Van Gogh-Goghs, you can have an account here.'], 
    ['Illegal content of any kind will not be tolerated. Any kind. Any. Kind. Don\'t make me go research what this means in every jurisdiction in the world. it\'s easier to just ban you.'],
    ['Only people who are 18+ may join'], 
    ['Use content warnings and appropriate tags for any NSFW / Lewd content'], 
    ['Be excellent to each other. No racism, homophobia, ageism, classism, ableism, anti-semitism, etc. will be tolerated.'], 
    ['No illegal content allowed'], 
    ['If you are able, add sources to image / video posts. Giving models / creators credit is a good thing to do.'], 
    ['No spam / advertising / crap like that'], 
    ['If you are trying to register to join this server, I will consider adding you if we have spoken before and you can point me to an account of yours I\'ve interacted with. Everyone else will be denied sight unseen.'], 
    ['No under 18 anything. Real, fake, written fantasy, nothing like that will be allowed here. This is a place for people to post 18+ content. No exceptions.'],
    ['Serverstandort: Deutschland. EU-Recht und deutsche Gesetze gelten.'], 
    ['Beiträge, für die die noAFD oder noSVP applaudieren würden, sind hier unerwünscht. Also keine faschistischen, rassistischen, sexistischen, queerfeindlichen und sonstigen entmenschlichenden Inhalte.'],
    ['No racism, sexism, homophobia, transphobia, ableism, or other discriminatory or hateful content.'], 
    ['No harassment, dogpiling, doxxing, or other personal attacks.'], 
    ['No hate speech or promotion of violence, white supremacy, antisemitism, or other hateful ideologies.'], 
    ['No pornography or depictions of sexual violence.'],
    ['No deliberately antagonistic, inflammatory posting meant to provoke angry and emotional responses. (trolling/shitposting)'], 
    ['No posts promoting unsupported conspiracy theories, or other false or misleading information.'], 
    ['No content illegal in the United States.'],
    ['Sexually explicit or violent media must be marked as sensitive when posting'], 
    ['No harassment, dogpiling or doxxing of other users'], 
    ['Do not share intentionally false or misleading information'], 
    ['Be respectful regarding the expression of different worldviews'], 
    ['No content illegal in Germany'],
    ['No racism, sexism, transphobia, or xenophobia.'], 
    ['CW all nudity and gore'], 
    ['No nazis, No fash, no cops/feds, no sexual depictions of children.'], 
    ['No stalking or harassment.'], 
    ['Ad policy: Music promotion is allowed, but ONLY 2 posts per show. Do not spam.'], 
    ['Jeff Bezos is not allowed on this server. Do not let him in. This is a bannable offense.'],
    ['Be excellent to each other.'], 
    ['@chris is the boss. His rule is LAW.'], 
    ['Don\'t post anything that will get us de-federated.'], 
    ['Don\'t be an asshole.'],
    ['We do not tolerate threatening behaviour, stalking, and doxxing.'], 
    ['We do not tolerate discriminatory behaviour and content promoting or advocating the oppression of members of marginalized groups.'], 
    ['We do not tolerate harassment, including brigading, dogpiling, or any other form of contact with a user who has stated that they do not wish to be contacted.'], 
    ['We do not tolerate mobbing, including name-calling, intentional misgendering or deadnaming.'], 
    ['We do not tolerate nationalist propaganda, Nazi symbolism or promoting the ideology of National Socialism.'], 
    ['We do not tolerate conspiracy narratives or other reactionary myths supporting or leading to the above-mentioned (and/or similar) behavior.'], 
    ['Actions intended to damage this instance or its performance may lead to immediate account suspension.'], 
    ['Content that is illegal in Germany will be deleted and may lead to immediate account suspension.'], 
    ['Use of bots on this instance is subject to admins\' authorization.'],
    ['DO: Post content that is meaningful to you'], 
    ['DO: Treat everyone on server and across servers with respect and decency'], 
    ['DO: Support claims with clear evidence from reputable sources'], 
    ['DO: Use content warnings when discussing sensitive topics (ex. violence)'], 
    ['DON\'T: Be tolerant of intolerance - report and disengage from people or servers that do not treat all people with respect and decency'], 
    ['DON\'T: Stalk, harass, or dox anyone'], 
    ['DON\'T: Post gore, graphic violence, pedophilia, necrophilia, etc.'], 
    ['DON\'T: Discriminate against others (ex. no racism, sexism, xenophobia, homophobia, transphobia, etc.)'], 
    ['DON\'T: Run commercial accounts or post advertisements for business services'], 
    ['DON\'T: Spam people with unwanted or repetitive content (ex. no bots)']
]




# Generate embeddings for all rules
embeddings = []
for rule in rules_data:
    rule_text = ' '.join(rule)
    # Tokenize input text
    input_ids = tokenizer.encode(rule_text, add_special_tokens=True, return_tensors='pt')

    # Get BERT embeddings for [CLS] token
    with torch.no_grad():
        outputs = model(input_ids)
        cls_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(cls_embedding)

# Convert embeddings to numpy array
embeddings = np.array(embeddings)

# Perform t-SNE dimensionality reduction
tsne = TSNE(n_components=2, perplexity=5, random_state=42)  # Adjust perplexity
embeddings_tsne = tsne.fit_transform(embeddings)

# Add hover-over text using mplcursors
mplcursors.cursor(hover=True).connect(
    "add", lambda sel: sel.annotation.set_text(rules_data[sel.target.index])
)

# Plot the t-SNE visualization
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1])
# for i, txt in enumerate(rules_data):
#     plt.annotate(' '.join(txt), (embeddings_tsne[i, 0], embeddings_tsne[i, 1]))
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Visualization of BERT Embeddings')
plt.show()
