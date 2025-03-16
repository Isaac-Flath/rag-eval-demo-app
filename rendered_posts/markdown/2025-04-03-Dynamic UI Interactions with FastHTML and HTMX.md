Isaac Flath

AboutBlogRSS

Subscribe

AboutBlogRSS

Subscribe

# Building Dynamic Web UIs with FastHTML and HTMX

A simpler approach to creating interactive web applications using Python and hypermedia

## Contents

The Problem with Web UI InteractivityWhat is FastHTML?Getting Started with FastHTML and HTMXWhat FastHTML and HTMX Will Do For YouSimple Show/Hide Toggle: Your First Interactive ComponentClick-to-Edit: Server-Driven FormsThe Surprising Insight: Server-Side Can Feel Like Client-SideReal-Time Updates and Advanced PatternsInfinite ScrollingPutting It All TogetherSimplified Development WorkflowEnhanced MaintainabilityKey Principles to RememberNext Steps for Your Journey

Modern web development requires complicated dependencies and extensive boilerplate spread over multiple languages to create interactive UIs. FastHTML with HTMX is here to fix that.

## The Problem with Web UI Interactivity

Building interactive web applications has always been complicated. I spent years banging my head against the wall with web development. My Python backend code was a thing of beauty - elegant, readable, maintainable. But the moment I needed a simple dropdown to update dynamically? Pure chaos.

The worst part? Most of this complexity was for ridiculously simple interactions.

Traditional approaches force you to choose between:

  1. Heavy JavaScript frameworks that take over your entire frontend
  2. Server-rendered pages with poor interactivity
  3. Complex hybrid approaches requiring expertise in multiple paradigms

There had to be a saner approach to building interactive web UIs

## What is FastHTML?

FastHTML is a Python library that makes generating HTML intuitive and pythonic. It allows you to:

  * Create HTML elements using Python functions (e.g., `Div()`, `Button()`)
  * Compose elements hierarchically using function composition
  * Add attributes using keyword arguments or chained methods
  * Integrate seamlessly with HTMX for interactivity

## Getting Started with FastHTML and HTMX

Before diving into examples, let's set up a minimal FastHTML application:

[code]

    from fasthtml.common import *
    
    @rt
    def index():
        return Div(
            H1("Hello, FastHTML!"),
            "This is my first FastHTML application.")
    
    serve()
    
[/code]

Save this as `app.py` and run with `python app.py`. Visit http://localhost:5001 to see your application.

> 💡 The `@rt` decorator (short for "route") is FastHTML's way of defining HTTP endpoints. It automatically:

  * Converts the function into a web route
  * Handles parameter parsing from requests
  * Renders the returned FastHTML elements to HTML

## What FastHTML and HTMX Will Do For You

FastHTML lets anyone build high-quality, interactive web apps in pure Python. HTMX is a small JavaScript library (< 14KB) that allows you to access modern browser features directly from HTML, rather than writing your own JavaScript. FastHTML is a Python library that makes it easy to generate HTML with HTMX attributes, letting you build dynamic interfaces with minimal code.

Together, they:

  * Bring simplicity to web interactivity development
  * Allow you to write your entire application in Python
  * Let you focus on building features instead of managing frontend state

> For comprehensive documentation see the FastHTML Documentation and HTMX Documentation

Let's see a simple example of a show/hide toggle:

[code]

    def mk_button(show):
        return Button("Hide" if show else "Show",
            hx_get="toggle?show=" + ("False" if show else "True"),
            hx_target="#content", id="toggle", hx_swap_oob="outerHTML")
    
    @rt
    def index(): return Div(mk_button(False), Div(id="content"))
    
    @rt
    def toggle(show: bool):
        return Div(
            Div(mk_button(show)),
            Div("Content that is toggled!" if show else ''))
    
[/code]

> 💡 Tip You can follow the example links to see full running examples and the source code you can use to try it out yourself locally!

With just these few lines of Python, I've created a toggle button that shows and hides content without writing a single line of JavaScript. The `hx_get` attribute tells HTMX to make a GET request to the `toggle` endpoint when clicked, and the `hx_target` attribute tells it to replace the element with ID "content" with the response.

By the end of this post, you'll be able to implement complex UI patterns like inline validation, infinite scrolling, and even real-time chat - all with a fraction of the code you'd need with traditional approaches.

> 💡 When to Consider Alternatives
>
>   * **Highly Interactive UIs** : Applications requiring complex client-side state management (like graphic editors)
>   * **Offline-First Applications** : Apps that need to function without network connectivity
>   * **Extremely High-Performance Needs** : Cases where minimizing network roundtrips is critical (this is not most cases)
>

# Simple Show/Hide Toggle: Your First Interactive Component

Let's start our journey with a simple but powerful pattern: the show/hide toggle. This is a common UI element that allows users to expand and collapse content, saving screen space and reducing visual clutter.

[code]

    def mk_button(show):
        return Button(
            "Hide" if show else "Show",
            hx_get="toggle?show=" + ("False" if show else "True"),
            hx_target="#content", id="toggle", 
            hx_swap_oob="outerHTML")
    
    @rt
    def index():
        return Div(mk_button(False), Div(id="content"))
    
    @rt
    def toggle(show: bool):
        return Div(
            Div(mk_button(show)),
            Div("Content that is toggled!" if show else ''))
    
[/code]

This code creates a simple page with a button that toggles the visibility of a block of text. Let's break down how it works:

  1. The `mk_button` function creates a button with HTMX attributes:

     * `hx_get` specifies the endpoint to call when clicked
     * `hx_target` identifies which element to update with the response
     * `hx_swap_oob` ("out of band") allows us to update multiple elements at once
  2. The `index` route renders the initial state with a "Show" button and an empty content div

  3. The `toggle` route handles the button click:

     * It takes a boolean parameter indicating whether to show content
     * It returns a new button (toggled state) and the content div (filled or empty)

When a user clicks the button, HTMX makes a GET request to our server, which responds with HTML that updates both the button and content area. The entire interaction happens without a page refresh or any custom JavaScript.

## How It Works Behind the Scenes

What makes this approach powerful is the seamless integration between the client and server. Here's what happens when a user clicks the button:

  1. HTMX intercepts the click event
  2. It makes an HTTP request to `/toggle?show=True`
  3. Our server processes the request and generates new HTML
  4. HTMX replaces the targeted elements with the new HTML

This pattern - intercepting events, making HTTP requests, and updating the DOM - is the foundation for all HTMX interactions.

But what happens when we need more complex interactions? Let's explore that next.

# Click-to-Edit: Server-Driven Forms

One of the most common UI patterns in modern web applications is the "click-to-edit" pattern. This allows users to view data in a clean, readable format, then click to transform it into an editable form. When they're done editing, they submit the changes, and the view returns to its original state with updated data. With FastHTML and HTMX, we can implement it with pure Python and a few HTMX attributes.

Here's how we can implement a click-to-edit pattern using FastHTML and HTMX:

[code]

    flds = dict(firstName='First Name', lastName='Last Name', email='Email')
    
    @dataclass
    class Contact:
        firstName:str; lastName:str; email:str; edit:bool=False
        def __ft__(self):
            "The __ft__ method determines how a `Contact` is rendered and displayed"
            def item(k, v):
                val = getattr(self,v)
                return Div(Label(Strong(k), val), Hidden(val, id=v))
            return Form(
                *(item(v,k) for k,v in flds.items()),
                Button('Click To Edit'),
                post='form', hx_swap='outerHTML')
    
    contacts = [Contact('Joe', 'Blow', 'joe@blow.com')]
    
    @rt
    def index(): return contacts[0]
    
    @rt
    def form(c:Contact):
        def item(k,v): return Div(Label(k), Input(name=v, value=getattr(c,v)))
        return Form(
            *(item(v,k) for k,v in flds.items()),
            Button('Submit', name='btn', value='submit'),
            Button('Cancel', name='btn', value='cancel'),
            post="contact", hx_swap='outerHTML'
        )
    
    @rt
    def contact(c:Contact, btn:str):
        if btn=='submit': contacts[0] = c
        return contacts[0]
    
[/code]

This code creates a contact information display that transforms into an editable form when clicked. Let's break down how it works:

  1. We define a `Contact` class with a custom `__ft__` method that determines how it renders in FastHTML
  2. In view mode, it displays formatted contact information with a "Click To Edit" button
  3. When clicked, HTMX sends a POST request to the `form` endpoint
  4. The `form` endpoint returns an editable form with the contact's current data
  5. When submitted, HTMX sends the form data to the `contact` endpoint
  6. The `contact` endpoint updates the data and returns to view mode

The entire interaction happens without a page refresh, giving users a smooth, app-like experience—all with server-rendered HTML.

## Technical Principles at Work

This example demonstrates several powerful principles:

  1. **HTML as the Transport Layer** : We're sending HTML fragments between client and server, not JSON
  2. **Server-Side State Management** : The server maintains the state of the application
  3. **Targeted DOM Updates** : HTMX only updates the parts of the page that need to change
  4. **Declarative Behavior** : The behavior is declared in HTML attributes, not JavaScript code

## Real-Time Form Validation

While our click-to-edit example is already powerful, we can take it a step further with real-time form validation. This is where HTMX truly shines—allowing us to validate form inputs as users type, providing immediate feedback without writing any JavaScript.

Let's see how we can implement real-time validation for a more complex form:

[code]

    @rt
    def index():
        return Form(
            Div(Label('Email Address', _for='email'),
                Input(type='text', name='email', id='email', post='email'),
                hx_target='this', hx_trigger='changed', hx_swap='outerHTML'),
            Div(Button('Submit', type='submit', id='submit-btn'),
                id='submit-btn-container'),
                hx_post=submit, hx_target='#submit-btn-container', hx_swap='outerHTML')
    
    @rt
    def email(email: str):
        error_msg = validate_email(email)
        return Div(
            Label('Email Address'),
            Input(name='email', type='text', value=f'{email}', post='email'),
            Div(f'{error_msg}', style='color: red;') if error_msg else None,
            hx_target='this', 
            hx_swap='outerHTML', cls=f"{error_msg if error_msg else 'Valid'}")
    
    @rt
    def submit(email: str):
        errors = {'email': validate_email(email)}
        errors = {k: v for k, v in errors.items() if v is not None}
        return Div(
            Button("Submit", type='submit', id='submit-btn'),
            *[Div(error, style='color: red;') for error in errors.values()],
            id='submit-btn-container')
    
    def validate_email(email: str):
        email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if not re.match(email_regex, email):
            return "Please enter a valid email address"
        return None
    
[/code]

This example creates a form with an email fields, validated in real-time. Let's break down how it works:

  1. Each input field has a `post` attribute that triggers a request to a validation endpoint when the value changes
  2. The validation endpoint checks the input and returns HTML with appropriate error messages
  3. HTMX replaces the input field with the validated version, showing any error messages
  4. When the form is submitted, all fields are validated again on the server

The result is a form that provides immediate feedback to users, guiding them toward valid input without any custom JavaScript. This is a powerful pattern that can be applied to any form in your application.

## The Surprising Insight: Server-Side Can Feel Like Client-Side

One of the most counterintuitive aspects of this approach is that server-side validation can feel just as responsive as client-side validation. By making small, targeted requests and updating only the necessary parts of the page, HTMX creates an experience that feels instant to users.

In the next section, we'll explore even more powerful patterns, including loading indicators, infinite scrolling, and real-time updates with WebSockets.

# Real-Time Updates and Advanced Patterns

Now that we've explored basic interactions and form validation, let's look at more sophisticated patterns that truly showcase the power of FastHTML and HTMX together.

## Infinite Scrolling

Infinite scrolling is a pattern that automatically loads more content as the user scrolls down a page. With HTMX, it's remarkably simple compared to other alternatives.

[code]

    column_names = ('name', 'email', 'id')
    
    def generate_contact(id: int) -> Dict[str, str]:
        return {'name': 'Agent Smith',
                'email': f'void{str(id)}@matrix.com',
                'id': str(uuid.uuid4())}
    
    def generate_table_row(row_num: int) -> Tr:
        contact = generate_contact(row_num)
        return Tr(*[Td(contact[key]) for key in column_names])
    
    def generate_table_part(part_num: int = 1, size: int = 20) -> Tuple[Tr]:
        paginated = [generate_table_row((part_num - 1) * size + i) for i in range(size)]
        paginated[-1].attrs.update({
            'get': f'page?idx={part_num + 1}',
            'hx-trigger': 'revealed',
            'hx-swap': 'afterend'})
        return tuple(paginated)
    
    @rt
    def index():
        return Titled('Infinite Scroll',
                      Div(Table(
                          Thead(Tr(*[Th(key) for key in column_names])),
                          Tbody(generate_table_part(1)))))
    
    @rt
    def page(idx:int|None = 0):
        return generate_table_part(idx)
    
[/code]

The magic happens with the `hx_trigger="revealed"` attribute, which fires when an element becomes visible in the viewport. When the loading indicator is revealed, HTMX automatically fetches more content and seamlessly adds it to the page.

## WebSockets for Real-Time Updates

WebSockets provide a persistent connection between client and server, enabling real-time updates without polling. Setting up WebSockets with FastHTML requires:

  1. Adding the WebSocket extension to your application
  2. Creating WebSocket handlers for connection events
  3. Using the `ws_connect` attribute to establish connections

[code]

    # All messages here, but only most recent 15 are stored
    messages = deque(maxlen=15)
    users = {}
    
    # Takes all the messages and renders them
    box_style = "border: 1px solid #ccc; border-radius: 10px; padding: 10px; margin: 5px 0;"
    def render_messages(messages):
        return Div(*[Div(m, style=box_style) for m in messages], id='msg-list')
    
    # Input field is reset via hx_swap_oob after submitting a message
    def mk_input(): return Input(id='msg', placeholder="Type your message", value="", hx_swap_oob="true")
    
    @rt
    def index():
        return Titled("Leave a message for others!"),Div(
            Form(mk_input(), ws_send=True), # input field
            P("Leave a message for others!"),
            Div(render_messages(messages),id='msg-list'), # All the Messages
            hx_ext='ws', ws_connect='ws') # Use a web socket 
    
    def on_connect(ws, send): users[id(ws)] = send
    def on_disconnect(ws):users.pop(id(ws),None)
    
    @app.ws('/ws', conn=on_connect, disconn=on_disconnect)
    async def ws(msg:str,send):
        await send(mk_input()) # reset the input field immediately
        messages.appendleft(msg) # New messages first
        for u in users.values(): # Get `send` function for a user
            await u(render_messages(messages)) # Send the message to that user
    
[/code]

With just these few lines of code, we've created a real-time chat application where messages are instantly broadcast to all connected clients. No JavaScript frameworks, no complex state management, just HTML and Python.

# Putting It All Together

Let's step back and consider what we've accomplished. We've built:

  1. A simple show/hide toggle
  2. A click-to-edit interface with form validation
  3. An infinite scrolling content loader
  4. A real-time chat application

All with minimal code, no JavaScript, and a unified development experience in Python. This approach brings several key benefits:

## Simplified Development Workflow

With FastHTML and HTMX, your entire application lives in Python. This means:

  * One language to learn and maintain
  * Consistent patterns throughout your codebase
  * No context switching between frontend and backend
  * Easier debugging and testing

## Enhanced Maintainability

As your application grows, the benefits become even more apparent:

  * Clear separation of concerns
  * Predictable request/response patterns
  * Server-side rendering for better SEO
  * Simplified state management (the server is the source of truth)

# Key Principles to Remember

As you start building with FastHTML and HTMX, keep these principles in mind:

  1. **Think in HTML, not JavaScript** : Focus on the HTML structure and how it changes, not on client-side logic.

  2. **Embrace hypermedia** : Use links, forms, and HTMX attributes to create interactive applications without custom JavaScript.

  3. **Server as the state manager** : Let your server handle state transitions and business logic, returning HTML that reflects the new state.

  4. **Progressive enhancement** : Start with functional HTML and enhance it with HTMX

# Next Steps for Your Journey

Now that you understand the basics of FastHTML and HTMX, here are concrete steps to continue your learning:

  1. **Start with a small project** : Convert an existing static page to use FastHTML
  2. **Explore the pattern gallery** : Visit the FastHTML gallery to see dozens of UI patterns implemented
  3. **Join the community** : FastHTML Discord
  4. **Contribute** : Both projects are open source and welcome contributions

What will you build with FastHTML and HTMX?

### Stay Updated

Get notified about new posts on AI, web development, and tech insights.

Subscribe

## Contents

The Problem with Web UI InteractivityWhat is FastHTML?Getting Started with FastHTML and HTMXWhat FastHTML and HTMX Will Do For YouSimple Show/Hide Toggle: Your First Interactive ComponentClick-to-Edit: Server-Driven FormsThe Surprising Insight: Server-Side Can Feel Like Client-SideReal-Time Updates and Advanced PatternsInfinite ScrollingPutting It All TogetherSimplified Development WorkflowEnhanced MaintainabilityKey Principles to RememberNext Steps for Your Journey

## Let's Connect

GitHub

LinkedIn

Twitter

© 2025 Isaac Flath • All rights reserved

