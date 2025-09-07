import requests
import time
import random


def post_with_retries(
    url,
    headers=None,
    data=None,
    timeout=30,
    max_retries=3,
    backoff_base=2,
):
    """
    Envoie une requête POST robuste avec retries et backoff exponentiel.

    Args:
        url (str): URL de l'API.
        headers (dict): En-têtes HTTP.
        data (bytes/str): Corps de la requête.
        timeout (int): Timeout max en secondes par tentative.
        max_retries (int): Nombre max de tentatives en cas d'échec.
        backoff_base (int): Facteur de base du délai exponentiel.

    Returns:
        requests.Response | None : Réponse si succès, sinon None.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, data=data, timeout=timeout)

            if response.status_code == 200:
                return response

            elif response.status_code == 429:
                # Backoff exponentiel + jitter
                wait = backoff_base**attempt + random.random()
                print(f"⚠️ 429 Too Many Requests → retry dans {wait:.1f}s")
                time.sleep(wait)

            elif 500 <= response.status_code < 600:
                # Erreurs serveur : possible de retenter
                wait = backoff_base**attempt
                print(f"⚠️ Erreur serveur {response.status_code}, retry dans {wait}s")
                time.sleep(wait)
            else:
                # Erreur client ou autre → inutile de retenter
                print(f"❌ Erreur {response.status_code}, abandon")
                return None

        except requests.exceptions.Timeout:
            print(f"⏳ Timeout tentative {attempt}/{max_retries}")
            if attempt < max_retries:
                time.sleep(backoff_base**attempt)
            else:
                return None

        except requests.exceptions.RequestException as e:
            print(f"🌐 Erreur réseau non récupérable : {e}")
            return None

    return None  # si toutes les tentatives ont échoué
