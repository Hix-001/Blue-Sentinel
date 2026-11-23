/**
 * Authentication & Security Clearance Manager for Blue Sentinel.
 * Manages JWT tokens, role-based states (OPERATOR, ANALYST, COMMANDER),
 * and the Tactical Access Terminal modal.
 */

export class AuthManager {
  constructor() {
    this.tokenKey = "sentinel_auth_token";
    this.userKey = "sentinel_auth_user";
    this.currentUser = null;
    this.baseUrl = import.meta.env.VITE_API_URL || "";

    this._initElements();
    this._attachEventListeners();
    this.checkSession();
  }

  _initElements() {
    this.modal = document.getElementById("auth-modal");
    this.form = document.getElementById("auth-form");
    this.usernameInput = document.getElementById("auth-username");
    this.passwordInput = document.getElementById("auth-password");
    this.errorMsg = document.getElementById("auth-error-msg");
    this.roleBadge = document.getElementById("operator-role-badge");
    this.authBtn = document.getElementById("btn-auth-trigger");
    this.closeBtn = document.getElementById("btn-close-auth");
  }

  _attachEventListeners() {
    if (this.authBtn) {
      this.authBtn.addEventListener("click", () => {
        if (this.currentUser) {
          this.logout();
        } else {
          this.openModal();
        }
      });
    }

    if (this.closeBtn) {
      this.closeBtn.addEventListener("click", () => this.closeModal());
    }

    // Quick-select preset buttons for 1-click credential test
    const quickBtns = document.querySelectorAll(".quick-cred-btn");
    quickBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const u = btn.getAttribute("data-user");
        const p = btn.getAttribute("data-pass");
        if (this.usernameInput) this.usernameInput.value = u;
        if (this.passwordInput) this.passwordInput.value = p;
        if (this.errorMsg) this.errorMsg.style.display = "none";
      });
    });

    if (this.form) {
      this.form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const username = this.usernameInput.value.trim();
        const password = this.passwordInput.value.trim();
        await this.login(username, password);
      });
    }
  }

  openModal() {
    if (this.modal) {
      this.modal.style.display = "flex";
      if (this.errorMsg) this.errorMsg.style.display = "none";
    }
  }

  closeModal() {
    if (this.modal) {
      this.modal.style.display = "none";
    }
  }

  getToken() {
    return localStorage.getItem(this.tokenKey);
  }

  async login(username, password) {
    if (this.errorMsg) this.errorMsg.style.display = "none";

    try {
      const res = await fetch(`${this.baseUrl}/api/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password })
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || "Authentication Failed");
      }

      const data = await res.json();
      localStorage.setItem(this.tokenKey, data.access_token);
      localStorage.setItem(this.userKey, JSON.stringify(data.user));
      this.currentUser = data.user;

      this._updateUI();
      this.closeModal();

      const ticker = document.getElementById("ticker-text");
      if (ticker) {
        ticker.innerText = `SECURITY CLEARANCE GRANTED // OPERATOR: ${data.user.callsign} [${data.user.role}]`;
      }
    } catch (err) {
      if (this.errorMsg) {
        this.errorMsg.innerText = `⚠️ ${err.message}`;
        this.errorMsg.style.display = "block";
      }
    }
  }

  logout() {
    localStorage.removeItem(this.tokenKey);
    localStorage.removeItem(this.userKey);
    this.currentUser = null;
    this._updateUI();

    const ticker = document.getElementById("ticker-text");
    if (ticker) {
      ticker.innerText = "SECURITY CLEARANCE REVOKED // UNVERIFIED MONITORING MODE";
    }
  }

  async checkSession() {
    const token = this.getToken();
    if (!token) {
      this._updateUI();
      return;
    }

    try {
      const res = await fetch(`${this.baseUrl}/api/auth/me`, {
        headers: { Authorization: `Bearer ${token}` }
      });

      if (res.ok) {
        const user = await res.json();
        this.currentUser = user;
      } else {
        this.logout();
      }
    } catch (e) {
      // Offline or network error: use cached user if present
      const cached = localStorage.getItem(this.userKey);
      if (cached) {
        try { this.currentUser = JSON.parse(cached); } catch (err) {}
      }
    }
    this._updateUI();
  }

  _updateUI() {
    if (!this.roleBadge || !this.authBtn) return;

    if (this.currentUser) {
      const role = this.currentUser.role || "OPERATOR";
      this.roleBadge.innerText = `${role} // ${this.currentUser.username || this.currentUser.sub}`;
      this.roleBadge.className = `pill-val role-${role.toLowerCase()}`;
      this.authBtn.innerText = "LOGOUT";
      this.authBtn.classList.add("active");
    } else {
      this.roleBadge.innerText = "UNVERIFIED";
      this.roleBadge.className = "pill-val unverified";
      this.authBtn.innerText = "LOGIN";
      this.authBtn.classList.remove("active");
    }
  }
}
