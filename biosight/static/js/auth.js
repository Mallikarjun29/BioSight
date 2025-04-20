document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('loginForm');
    const registerForm = document.getElementById('registerForm');

    if (loginForm) {
        loginForm.addEventListener('submit', handleLogin);
    }

    if (registerForm) {
        registerForm.addEventListener('submit', handleRegister);
    }
});

// Remove localStorage token handling functions since we're using cookies now

async function handleLogin(e) {
    e.preventDefault();
    clearErrors();

    // Use FormData to work with OAuth2PasswordRequestForm on backend
    const formData = new FormData();
    formData.append('username', document.getElementById('email').value); // OAuth2 expects 'username'
    formData.append('password', document.getElementById('password').value);

    try {
        const response = await fetch('/api/login', {
            method: 'POST',
            body: formData, // FormData for OAuth2PasswordRequestForm
        });

        if (response.ok) {
            // Cookie is automatically set by the server
            // Just redirect to home page
            window.location.replace('/');
        } else {
            let errorDetail = 'Login failed';
            try {
                const errorData = await response.json();
                errorDetail = errorData.detail || errorDetail;
            } catch (e) {
                // If response isn't JSON, use default error
            }
            showError('emailError', errorDetail);
        }
    } catch (error) {
        console.error('Login error:', error);
        showError('emailError', 'An unexpected error occurred during login.');
    }
}

async function handleRegister(e) {
    e.preventDefault();
    clearErrors();

    // Check password confirmation
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirmPassword').value;
    
    if (password !== confirmPassword) {
        showError('confirmPasswordError', 'Passwords do not match');
        return;
    }

    const userData = {
        email: document.getElementById('email').value,
        password: password,
        name: document.getElementById('name').value
    };

    try {
        const response = await fetch('/api/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(userData),
        });

        if (response.ok) {
            // Registration successful, login automatically
            const loginFormData = new FormData();
            loginFormData.append('username', userData.email); 
            loginFormData.append('password', userData.password);
            
            const loginResponse = await fetch('/api/login', {
                method: 'POST',
                body: loginFormData,
            });

            if (loginResponse.ok) {
                window.location.replace('/');
            } else {
                window.location.href = '/login?registered=true';
            }
        } else {
            let errorDetail = 'Registration failed';
            try {
                const errorData = await response.json();
                errorDetail = errorData.detail || errorDetail;
            } catch (e) {
                // If response isn't JSON, use default error
            }
            showError('emailError', errorDetail);
        }
    } catch (error) {
        console.error('Registration error:', error);
        showError('emailError', 'An error occurred during registration');
    }
}

function showError(elementId, message) {
    const errorElement = document.getElementById(elementId);
    if (errorElement) {
        errorElement.textContent = message;
    }
}

function clearErrors() {
    const errorElements = document.querySelectorAll('.error-message');
    errorElements.forEach(element => {
        element.textContent = '';
    });
}

// Remove localStorage check functions - cookies are handled automatically by browser